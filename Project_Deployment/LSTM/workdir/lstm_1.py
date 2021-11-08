import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from sklearn import preprocessing
from sklearn.metrics import r2_score


# v2：读数据target格式有变化，现在target传入的是一个列表，返回结果也多了一个列名，传入的csv第一行必须为表头
# 读取path中的csv，转换成一个2维的1*n的array数组，可做兼容，读取其他类型的文件，如xls
# path 表示读取文件路径
# target表示读取的列数，从0开始，默认为0
def read_data(path, target=[0]):
    dataframe = pd.read_csv(path, usecols=target, engine='python')
    dataset = dataframe.values

    columns = dataframe.columns
    # 将整型变为float
    dataset = dataset.astype('float32')
    print(columns)
    return dataset, columns


# 数据预处理，归一化
# normalization表示归一化的方式，1为MinMaxScaler，2为StandardScaler，0为不做归一化处理，默认为0
# dataset是一个2维的1*n的array数组
def data_preprocessing(dataset, normalization=0):
    if normalization == 1:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif normalization == 2:
        scaler = preprocessing.StandardScaler()
    else:
        return dataset, None
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


def split_train_test(dataset, rate=1):
    train_size = int(len(dataset) * rate)
    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]
    return trainlist, testlist


def create_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# V2
def build_LSTM(trainX, trainY, formation=[64], epochs=100, batch_size=None, task_id=None):
    model = Sequential()
    for i in range(len(formation)):
        if i < (len(formation) - 1):
            model.add(LSTM(formation[i], input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        else:
            model.add(LSTM(formation[i], input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))

    model.compile(loss='mean_squared_error', optimizer='adam')
    if batch_size is None:
        batch_size = len(trainX)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    model.save(os.path.join("LSTM_model" + ".h5"))
    return model


# v2
# 传入一个一维数组，长度与时间戳相等，expand_num表示向后预测expand_num个时间点
def predict_with_LSTM(dataset, look_back, scaler=None, task_id=None, model=None, expand_num=1):
    if task_id is not None:
        model = load_model(os.path.join("LSTM_model" + ".h5"))
    elif model is not None:
        model = model
    else:
        print("请输入训练好的模型")
    new_dataset = numpy.reshape(dataset, (1, look_back, int(len(dataset) / look_back)))

    data_predict = []

    for i in range(expand_num):
        predict_value = model.predict(new_dataset)
        data_predict.extend(predict_value.tolist())
        new_dataset = numpy.reshape(new_dataset, (len(dataset)))
        new_dataset = numpy.delete(new_dataset, [j for j in range(int(len(dataset) / look_back))])
        new_dataset = numpy.append(new_dataset, predict_value)
        new_dataset = numpy.reshape(new_dataset, (1, look_back, int(len(dataset) / look_back)))
    if scaler is not None:
        pass
        data_predict = scaler.inverse_transform(data_predict)
    return data_predict


# v2
def predict_with_curData(dataset, scaler=None, task_id=None, model=None):
    if task_id is not None:
        model = load_model(os.path.join("LSTM_model" + ".h5"))
    elif model is not None:
        model = model
    else:
        print("请输入训练好的模型")
    data_predict = model.predict(dataset)
    if scaler is not None:
        data_predict = scaler.inverse_transform(data_predict)
        print(data_predict)
    return data_predict.tolist()


# v2:参数需要额外传入一个保存的文件名
def draw(raw_data, predict_data, picture_name, yname):
    plt.plot(raw_data)
    plt.plot(predict_data)
    # plt.show()
    plt.xlabel("index")
    plt.ylabel(yname)
    plt.savefig(picture_name)
    plt.clf()


# v2:index_col传参类型变化，变成了列表
# path: 读取文件的路径
# look_back: 选择的划分时间戳的长度
# index_col: 读取文件的哪一列，默认读取第一列
# normalization：表示归一化的方式，1为MinMaxScaler，2为StandardScaler，0为不做归一化处理，默认为0
# formation：构建的LSTM网络的结构，是一个一维数组，默认是一个隐藏层，层数是8
# epochs：迭代次数
# batch_size：每次训练放入的数据集批次，当不存在时，默认以训练集的个数
# expand_num：向后预测多少个点，默认为1
def LSTM_module(path, look_back, index_col=[0], normalization=0, formation=[8], epochs=100, batch_size=None,
                expand_num=1):
    dataset, columns = read_data(path, index_col)
    # dataset = read_data('./flights.csv', [1])
    dataset, scaler = data_preprocessing(dataset, normalization=normalization)
    trainX, trainY = create_dataset(dataset, look_back)
    model = build_LSTM(trainX, trainY, formation=formation, epochs=epochs, batch_size=batch_size)
    test_set = []
    lenth = len(dataset)
    for i in range(look_back):
        for i in dataset[lenth - (look_back - i), :]:
            test_set.append(float(i))
    data_predict = predict_with_LSTM(test_set, look_back, scaler=scaler, model=model, expand_num=expand_num)
    dataset_predict = predict_with_curData(trainX, scaler=scaler, model=model)
    if scaler is not None:
        trainY = scaler.inverse_transform(trainY)
    r2_list = []
    for i in range(len(index_col)):
        r2 = r2_score(trainY[:, i], numpy.array(dataset_predict)[:, i])
        r2_list.append(r2)

    picture_list = []
    for i in data_predict:
        dataset_predict.append(i)
    for i in range(len(index_col)):
        picture_name = "result" + str(i) + ".jpg"
        draw(trainY[:, i], numpy.array(dataset_predict)[:, i], picture_name, columns[i])
        picture_list.append(picture_name)
    numpy.savetxt('result.csv', data_predict, delimiter=',')

    return r2_list, picture_list,columns


if __name__ == "__main__":
    r2_list, picture_list, columns = LSTM_module('./test.csv', 3, index_col=[1,2,3], normalization=0, formation=[8], epochs=100,
                                        batch_size=1,
                                        expand_num=10)
    print(r2_list, picture_list, columns)
