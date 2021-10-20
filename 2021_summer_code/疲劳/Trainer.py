import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random

import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，忽略所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 忽略 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 忽略 Error

from sklearn.model_selection import KFold
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import math
from matplotlib import pyplot as plt
import numpy as np
from Read_csv import get_features_targets

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)


def liner_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: (((x - min_val) / scale) - 1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: (
        min(max(x, clip_to_min), clip_to_max)
    ))


def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1.0))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


def normalize_scale(examples_dataframe):
    '''
    normalize dataframe data
    :param examples_dataframe: input dataframe
    :return: output dataframe
    '''
    processed_features = pd.DataFrame()
    not_linear_features = ["melting_method", "hot_working_mode", "hot_working_process", "hot_working_state",
                           "hot_working_state_1"]
    for feature in examples_dataframe:
        if feature in not_linear_features:
            # continue
            processed_features[feature] = liner_scale(examples_dataframe[feature])
        else:
            processed_features[feature] = log_normalize(examples_dataframe[feature])
    return processed_features


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}
    # targets = {key: np.array(value) for key,value in dict(targets).items()}
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(100)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        features,
        targets):
    '''
    Train a model to predict target and output graphs to show the result.
    :param my_optimizer: optimizer
    :param steps: sum steps
    :param batch_size: each batch has the number of data
    :param hidden_units: hidden layers
    :param features: input features
    :param targets: input targets
    :return: A trained model
    '''

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(features),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        model_dir='./models_' + target_name
    )

    # define variables which used in loop
    periods = 10
    steps_per_period = steps / periods
    period = 0
    training_root_mean_squared_error = 0
    validation_root_mean_squared_error = 0
    training_R2 = 0
    validation_R2 = 0
    training_rmse = []
    validation_rmse = []
    training_R2_list = []
    validation_R2_list = []

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    kf = KFold(n_splits=periods, shuffle=True, random_state=1)
    print("Training model...")
    print("Regression evaluation index (on training data):")

    for train_index, test_index in kf.split(features):
        period += 1
        print("TRAIN:", train_index)
        print("TEST:", test_index)
        training_examples = features.iloc[train_index, :]
        training_targets = targets.iloc[train_index, :]
        validation_examples = features.iloc[test_index, :]
        validation_targets = targets.iloc[test_index, :]
        # Create input functions.
        training_input_fn = lambda: my_input_fn(training_examples,
                                                training_targets[target_name_cn],
                                                batch_size=batch_size)
        predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                        training_targets[target_name_cn],
                                                        num_epochs=1,
                                                        shuffle=False)
        predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                          validation_targets[target_name_cn],
                                                          num_epochs=1,
                                                          shuffle=False)

        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets[target_name_cn]))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets[target_name_cn]))

        training_R2 = metrics.r2_score(training_predictions, training_targets[target_name_cn])
        validation_R2 = metrics.r2_score(validation_predictions, validation_targets[target_name_cn])
        # Occasionally print the current loss.
        print("\033[0;31m Training RMSE  period %02d : %0.2f \033[0m" % (period, training_root_mean_squared_error))
        print("\033[0;31m Training R2  period %02d : %0.2f \033[0m " % (period, training_R2))
        print("\033[0;31m Validation RMSE  period %02d : %0.2f \033[0m" % (period, validation_root_mean_squared_error))
        print("\033[0;31m Validation R2  period %02d : %0.2f \033[0m " % (period, validation_R2))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        training_R2_list.append(training_R2)
        validation_R2_list.append(validation_R2)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.figure(1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.figure(2)
    plt.ylabel("R2")
    plt.xlabel("Periods")
    plt.title("R-Square vs. Periods")
    plt.tight_layout()
    plt.plot(training_R2_list, label="training")
    plt.plot(validation_R2_list, label="validation")
    plt.legend()

    print("\033[0;31mFinal RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
    print("Final R2 (on training data):   %0.2f" % training_R2)
    print("Final R2 (on validation data): %0.2f" % validation_R2)
    average_R2 = sum(validation_R2_list) / periods
    print("Average R2 (on validation data): %0.2f \033[0m " % average_R2)
    plt.show()
    return dnn_regressor


# target_name_cn = '屈服强度'
# target_name = 'yield_strength'

# target_name_cn = '延伸率'
# target_name = 'elongation'
# 
# target_name_cn = '抗拉强度'
# target_name = 'Tensile_strength'
# 
target_name_cn = '断面收缩率'
target_name = 'Reduction_of_section'

if __name__ == '__main__':
    # Choose the first 70 (out of 81) examples for training.
    features, targets = get_features_targets()

    # draw hist of features
    # features.hist(bins=20, figsize=(18, 12), xlabelsize=2)
    # plt.show()

    normalized_dataframe = normalize_scale(features)

    # draw hist of normalized features
    # normalized_dataframe.hist(bins=20, figsize=(18, 12), xlabelsize=2)
    # plt.show()

    # normalized_dataframe.to_csv('normalized_data.csv')
    # targets.to_csv('targets.csv')
    print(normalized_dataframe)
    _ = train_nn_regression_model(
        my_optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        steps=5000,
        batch_size=8,
        hidden_units=[32, 64, 32],
        features=normalized_dataframe,
        targets=targets
    )
