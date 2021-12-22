import os

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
import sys


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
    for i in range(len(dataset) - look_back):
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
    plt.title("Prediction value VS Actrual value")
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

    return r2_list, picture_list, data_predict, columns


def _read_parameters():
    import json
    def re_input(original: str):
        return original.replace('，', ',').replace(' ', '')
    def split2int(original: str):
        return list(map(int, original.split(',')))

    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    parameters['look_back'] = int(parameters['look_back'])
    parameters['start_index'] = int(parameters['start_index'])
    parameters['end_index'] = int(parameters['end_index'])
    parameters['normalization'] = int(parameters['normalization'])
    try:
        parameters['formation'] = split2int(re_input(parameters['formation']))
    except:
        parameters['formation'] = [16,32,64,32,16]
    try:
        parameters['epochs'] = int(parameters['epochs'])
    except:
        parameters['epochs'] = 100
    try:
        parameters['batch_size'] = int(parameters['batch_size'])
    except:
        parameters['batch_size'] = 1
    try:
        parameters['expand_num'] = int(parameters['expand_num'])
    except:
        parameters['expand_num'] = 3

    return parameters


def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _add_info_xml(picture_names, r2_scores, predictions, columns) -> None:
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import os
    # Back up result.xml
    os.system('cp ./result.xml ./result_before.xml')  # Linux
    # os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("./result.xml")
    root = tree.getroot()
    output = tree.find('output')

    # Table
    element_result = ET.Element('Result')
    element_table = ET.Element('table')
    for index, pres in enumerate(predictions):
        element_row = ET.Element('row')
        element_index = ET.Element('index')
        element_index.text = str(index + 1)
        element_row.append(element_index)
        for column, pre in zip(columns, pres):
            element_column = ET.Element(column)
            try:
                element_column.text = str(round(pre, 3))
            except:
                element_column.text = str(pre)
            element_row.append(element_column)
        element_table.append(element_row)
    element_result.append(element_table)
    output.append(element_result)

    # R2
    element_R2 = ET.Element('R2')
    element_table_R2 = ET.Element('R2_table')
    for column, r2_score in zip(columns, r2_scores):
        element_row = ET.Element('row')
        element_column = ET.Element('column')
        element_value = ET.Element('value')
        element_column.text = column
        try:
            element_value.text = str(round(r2_score, 3))
        except:
            element_value.text = str(r2_score)
        element_row.append(element_column)
        element_row.append(element_value)
        element_table_R2.append(element_row)
    element_R2.append(element_table_R2)
    output.append(element_R2)

    # Picture
    element_pictures = ET.Element('Pictures')
    for picture_name in picture_names:
        element_picture = ET.Element('picture')
        element_file, element_name, element_url = ET.Element('file'), ET.Element('name'), ET.Element('url')
        element_name.text, element_url.text = picture_name, picture_name
        element_file.append(element_name)
        element_file.append(element_url)
        element_picture.append(element_file)
        element_pictures.append(element_picture)
    output.append(element_pictures)

    # Save
    __indent(root)
    ET.tostring(root, method='xml')

    tree.write('result.xml', encoding='utf-8', xml_declaration=True)
    with open('result.xml', 'r') as fp:
        lines = [line for line in fp]
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/LSTM.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))


def _add_error_xml(error_message, error_detail):
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import os
    # Back up result.xml
    # os.system('cp ./result.xml ./result_before.xml')  # Linux
    os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("./result.xml")
    root = tree.getroot()
    output = tree.find('output')
    element_Error = ET.Element('ERROR')

    element_error = ET.Element('error')
    element_error.text = error_message
    
    element_error_detail = ET.Element('detail')
    element_error_detail.text = error_detail

    element_Error.append(element_error)
    element_Error.append(element_error_detail)

    for child in list(output):
        output.remove(child)
    output.append(element_Error)
    
    # Save
    __indent(root)
    ET.tostring(root, method='xml')

    tree.write('result.xml', encoding='utf-8', xml_declaration=True)
    with open('result.xml', 'r') as fp:
        lines = [line for line in fp]
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/LSTM.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))

def main():
    try:
        parameters = _read_parameters()
        inputCSV = parameters['inputCSV']
        look_back = parameters['look_back']
        column_indexs = list(range(parameters['start_index'] - 1, parameters['end_index'], 1))# star from 1.
        formation = parameters['formation']
        epochs = parameters['epochs']
        batch_size = parameters['batch_size']
        expand_num = parameters['expand_num']
        normalization = parameters['normalization']
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        r2, picture_names, data_predict, columns = LSTM_module(path=inputCSV, look_back=look_back, index_col=column_indexs, 
                        normalization=normalization, formation=formation, epochs=epochs, batch_size=batch_size,
                        expand_num=expand_num)
    except Exception as e:
        _add_error_xml("LSTM Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        _add_info_xml(picture_names, r2, data_predict, columns)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')


if __name__ == "__main__":
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close