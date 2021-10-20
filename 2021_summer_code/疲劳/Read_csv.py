import numpy as np
import pandas as pd
import re


np.random.seed(1)


def preprocess_features(datas):
    datas = spare_elements(datas)
    selected_features = datas[
        ["C",
         "Mn",
         "Si",
         "P",
         "S",
         "Cr",
         "Mo",
         "V",
         "Ni",
         "Cu",
         "Al",
         "melting_method",
         "hot_working_mode",
         "hot_working_process",
         "hot_working_state",
         "hot_working_state_1"
         ]]
    processed_features = selected_features.copy()
    #turn all object to float
    for features in processed_features:
        processed_features[features] = processed_features[features].astype(dtype=float)
    return processed_features


def preprocess_targets(datas):
    selected_features = datas[
        [
         "屈服强度",
         "抗拉强度",
         "延伸率",
         "断面收缩率"
         ]]
    output_targets = selected_features.copy()
    for features in output_targets:
        output_targets[features] = output_targets[features].astype(dtype=float)
    return output_targets


def spare_elements(data):
    '''
    a func turn object to one hot code.
    :param data: dataframe read from csv
    :return: one-hot data
    '''
    #熔炼方式
    dict_melting_method = {}
    index_melting_method = 1
    #热加工方式
    dict_hot_working_mode = {}
    index_hot_working_mode = 1
    #热处理工艺
    dict_hot_working_process = {}
    index_hot_working_process = 1
    #热处理状态
    dict_hot_working_state = {}
    index_hot_working_state = 1
    #热处理状态.1
    dict_hot_working_state_1 = {}
    index_hot_working_state_1 = 1
    for index, row in data.iterrows():
        if row['其它'] != 0:
            for element in row['其它'].split('，'):
                num_index = re.search("\d", element).start()
                data.loc[index, element[0:num_index]] = float(element[num_index:])
        else:
            data.loc[index, 'Cu'] = 0
            data.loc[index, 'Al'] = 0
        if row['熔炼方式'] != 0:
            if dict_melting_method.get(row['熔炼方式'], index_melting_method) == index_melting_method:
                dict_melting_method[row['熔炼方式']] = index_melting_method
                data.loc[index, 'melting_method'] = dict_melting_method[row['熔炼方式']]
                index_melting_method += 1
            else:
                data.loc[index, 'melting_method'] = dict_melting_method[row['熔炼方式']]
        if row['热加工方式'] != 0:
            if dict_hot_working_mode.get(row['热加工方式'], index_hot_working_mode) == index_hot_working_mode:
                dict_hot_working_mode[row['热加工方式']] = index_hot_working_mode
                data.loc[index, 'hot_working_mode'] = dict_hot_working_mode[row['热加工方式']]
                index_hot_working_mode += 1
            else:
                data.loc[index, 'hot_working_mode'] = dict_hot_working_mode[row['热加工方式']]
        if row['热处理工艺'] != 0:
            if dict_hot_working_process.get(row['热处理工艺'], index_hot_working_process) == index_hot_working_process:
                dict_hot_working_process[row['热处理工艺']] = index_hot_working_process
                data.loc[index, 'hot_working_process'] = dict_hot_working_process[row['热处理工艺']]
                index_hot_working_process += 1
            else:
                data.loc[index, 'hot_working_process'] = dict_hot_working_process[row['热处理工艺']]
        if row['热处理状态'] != 0:
            if dict_hot_working_state.get(row['热处理状态'], index_hot_working_state) == index_hot_working_state:
                dict_hot_working_state[row['热处理状态']] = index_hot_working_state
                data.loc[index, 'hot_working_state'] = dict_hot_working_state[row['热处理状态']]
                index_hot_working_state += 1
            else:
                data.loc[index, 'hot_working_state'] = dict_hot_working_state[row['热处理状态']]
        if row["热处理状态.1"] != 0:
            if dict_hot_working_state_1.get(row["热处理状态.1"], index_hot_working_state_1) == index_hot_working_state_1:
                dict_hot_working_state_1[row["热处理状态.1"]] = index_hot_working_state_1
                data.loc[index, "hot_working_state_1"] = dict_hot_working_state_1[row["热处理状态.1"]]
                index_hot_working_state_1 += 1
            else:
                data.loc[index, "hot_working_state_1"] = dict_hot_working_state_1[row["热处理状态.1"]]

    return data



def get_features_targets():
    datas = pd.read_csv("datas_flushed.csv", encoding='gbk')
    datas.fillna(0, inplace=True)
    datas.replace('-', 0, inplace=True)
    # datas = datas.reindex(np.random.permutation(datas.index))
    features = preprocess_features(datas)
    targets = preprocess_targets(datas)
    return features,targets


if __name__ == '__main__':
    from sklearn.model_selection import KFold
    features,targets = get_features_targets()
    kf = KFold(n_splits=8,shuffle=True,random_state=1)
    for train_index, test_index in kf.split(features):
        print("TRAIN:", train_index, "TEST:", test_index)
        print(features.iloc[train_index, :])
        # X_train, X_test = features[train_index], features[test_index]
        # Y_train, Y_test = targets[train_index], targets[test_index]
        # print(X_train)
