import sys

import joblib
import numpy as np
import pandas as pd
import pmdarima as pm
from matplotlib import pyplot as plt

from utils import _add_error_xml, _add_info_xml, _read_parameters


def read_data(path, feature=None, target=[0]):
    dataframe = pd.read_csv(path, engine='python')
    dataset = dataframe.values
    columns = dataframe.columns
    # 将整型变为float
    dataset = dataset.astype('float32')
    if feature is None:
        feature_data = None
        feature_columns = None
    else:
        feature_data = dataset[:, feature]
        feature_columns = columns[feature]
    target_data = dataset[:, target]
    target_columns = columns[target]
    return feature_data, target_data, feature_columns, target_columns


def arima_fit(feature_data, target_data,
              seasonal=False, m=0, information_criterion='bic',
              start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, ):
    if feature_data is not None and len(feature_data) != len(target_data):
        raise TypeError("输入的特征和目标的行数不一致")
    arima = pm.auto_arima(target_data, X=feature_data, error_action='ignore', trace=True,
                          suppress_warnings=True, maxiter=5,
                          seasonal=seasonal, m=m, information_criterion=information_criterion, start_p=start_p, d=d,
                          start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q, return_valid_fits=True)
    pickle_tgt = "arima.pkl"
    joblib.dump(arima[0], pickle_tgt)
    return arima


def arima_predict(model_path, file_path=None, n_periods=3):
    arima = joblib.load(model_path)
    if file_path is not None:
        dataframe = pd.read_csv(file_path, engine='python')
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return arima.predict(X=dataset, n_periods=len(dataset))
    return arima.predict(n_periods=n_periods)


def arima_prdict_insample(arima, X=None):
    return arima.predict_in_sample(X)


def arima(path, feature=None, target=[0], model_path="arima.pkl", file_path=None, n_periods=3, seasonal=False, m=0, information_criterion='bic', start_p=2, d=None,
          start_q=2, max_p=5, max_d=2, max_q=5):
    feature_data, target_data, feature_columns, target_columns = read_data(
        path, feature=feature, target=target)
    arimas = arima_fit(feature_data, target_data,
                       seasonal=seasonal, m=m, information_criterion=information_criterion,
                       start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q)
    draw_bic_or_aic(arimas, information_criterion, seasonal=False)
    if feature_data is None:
        y_predict = arima_predict(
            model_path, file_path=None, n_periods=n_periods)
        draw_predict(arimas[0], target_data, y_predict, feature=None)
    else:
        y_predict = arima_predict(
            model_path, file_path=file_path, n_periods=n_periods)
        draw_predict(arimas[0], target_data, y_predict, feature=feature_data)
    result_data = pd.read_csv(file_path)
    result_data['prediction'] = y_predict
    result_data.to_csv('result.csv', index=False)
    return result_data


def draw_bic_or_aic(arimas, information_criterion='bic', seasonal=False):
    x = []
    y = []
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if information_criterion == 'bic':
        plt.title('bic results for each model searched by grid traversal')
    elif information_criterion == 'aic':
        plt.title('aic results for each model searched by grid traversal')
    else:
        raise ValueError('information_criterion只能传递aic和bic两个值')
    for arima in arimas:
        arima_dict = arima.to_dict()
        x_item = arima_dict['order'].__str__()
        if seasonal:
            x_item = x_item + arima_dict['seasonal_order'].__str__()
        if x_item not in x:
            x.append(x_item)
            if information_criterion == 'bic':
                y.append(arima_dict['bic'])
            elif information_criterion == 'aic':
                y.append(arima_dict['aic'])
        else:
            index = x.index(x_item)
            if information_criterion == 'bic' and arima_dict['bic'] < y[index]:
                y[index] = arima_dict['bic']
            elif information_criterion == 'aic' and arima_dict['aic'] < y[index]:
                y[index] = arima_dict['aic']

    plt.barh(x, y)  # 横放条形图函数 barh
    plt.savefig(f'{information_criterion}.jpg')
    plt.clf()


def draw_predict(arima, y, y_predict, feature=None):
    cur_predict = arima_prdict_insample(arima, feature)
    x = np.arange(len(cur_predict))
    plt.scatter(x, y, marker='x')
    plt.plot(x, cur_predict)
    x = []

    for i in range(len(y_predict)):
        x.append(len(cur_predict) + i)
    plt.plot(x, y_predict)

    plt.title('Actual test samples vs. forecasts')
    plt.savefig("result.jpg")
    plt.clf()


def main():
    try:
        parameters = _read_parameters()
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    try:
        result = arima(parameters['trainCSV'], feature=parameters['features'], target=parameters['target'],
                   file_path=parameters['testCSV'], seasonal=parameters['seasonal'],
                   m=parameters['m'], information_criterion=parameters['information_criterion'],
                   start_p=parameters['start_p'], d=parameters['d'],
                   start_q=parameters['start_q'], max_p=parameters['max_p'],
                   max_d=parameters['max_d'], max_q=parameters['max_q'])
    except Exception as e:
        _add_error_xml("arima Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return 
    try:
        _add_info_xml(picture_names=['result.jpg',f"{parameters['information_criterion']}.jpg"], result=result)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
  
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')



if __name__ == '__main__':
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close
