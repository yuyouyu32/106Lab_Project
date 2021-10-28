# -*-coding=utf-8-*-
import sys
from typing_extensions import ParamSpec

import pandas as pd
import numpy as np
import os

from pandas import DataFrame
from sklearn.mixture import GaussianMixture as GM_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def calculate_dis_of_tp_and_vp_single(single_p, batch_p):
    """计算单个样本点(N维)到空间内其它点的欧氏距离
    param:
        single_p: 单个点的坐标，（N,）
        batch_p:  其余点的坐标，(samples，N)
    return:
        res：numpy.ndarray,(samples,)
    """
    b_s = (batch_p - single_p) ** 2
    su = np.sum(b_s, axis=1).astype(float)
    res = np.sqrt(su)
    return res


def calculate_dis_of_tp_and_vp_batch(created_data, origin_data):
    """计算虚拟样本点和原始数据集的欧氏距离
    param:
        created_data: 虚拟样本点，（samples_vs, N）
        origin_data:  原始数据集，(samples_t，N)
    return:
        res：numpy.ndarray,(samples_vs,samples_t)
        res_sort_idx：numpy.ndarray,(samples_vs,samples_t)
    """
    res_sort_idx = np.zeros(shape=(len(created_data), len(origin_data)), dtype="int")
    res = np.zeros(shape=(len(created_data), len(origin_data)))
    for i in range(len(created_data)):
        res_temp = calculate_dis_of_tp_and_vp_single(created_data[i, :], origin_data)
        res[i] = res_temp
        res_temp_sort_idx = np.argsort(res_temp)
        res_sort_idx[i] = res_temp_sort_idx
    return res, res_sort_idx


def gmm_vsg(file_path,
            expand_number,
            feature_space_start_idx,
            feature_space_end_idx,
            target_name=None,
            result_retain=2,
            random_state=None,
            is_aic=True,
            n_components_start=1,
            n_components_step=1, *args, **kwargs):
    """基于高斯混合模型来生成虚拟样本
    必选参数
    ----------
    file_path : 原始数据的存放路径
    expand_number : 要扩充生成多少条样本数目
    feature_space_start_idx : 原始数据特征空间起始列数
    feature_space_end_idx : 原始数据特征空间结束列数

    可选参数
    ----------
    target_name ： 是否分特征空间和目标值
    result_retain ： 结果保留的位数
    random_state : 随机种子
    is_aic ： 使用aic还是bic优化模型
    n_components_start： GMM起始组件数目
    n_components_step： GMM组件数目递增值
    """
    # 1.读数据
    # 数据可视化

    if not os.path.exists(file_path):
        print("文件路径不正确，请重新输入！")
        sys.exit(-1)
    df_data = pd.read_csv(file_path, delimiter=',')
    pca1 = None
    pca2 = None
    col_names = None
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    if target_name is None:
        X = df_data.values
        col_name = list(df_data.columns.values)
        col_names = col_name
        pca_X = X.copy()
        pca1 = PCA(n_components=2)
        pca1.fit(pca_X)
        X_2D = pca1.transform(pca_X)
        plt.figure()
        plt.scatter(X_2D[:, 0], X_2D[:, 1], color='blue')
        plt.xlabel('PCA-Dimension1')
        plt.ylabel('PCA-Dimension2')
        plt.savefig('origin.jpg')
        # plt.show()
    else:
        X = df_data.values[:, feature_space_start_idx - 1:feature_space_end_idx]
        y = df_data[target_name].values
        tem_col = df_data.columns.values[feature_space_start_idx - 1:feature_space_end_idx]
        col_name = [i for i in tem_col]
        col_name.append(target_name)
        col_names = col_name
        pca_X = np.hstack((X, y.reshape(-1, 1)))
        pca2 = PCA(n_components=2)
        pca2.fit(pca_X)
        X_2D = pca2.transform(pca_X)
        plt.figure()
        plt.scatter(X_2D[:, 0], X_2D[:, 1], color='blue')
        plt.xlabel('PCA-Dimension1')
        plt.ylabel('PCA-Dimension2')
        plt.savefig('origin.jpg')
        # plt.show()

    # 2. 求GMM参数
    n_components_end = len(X)
    n_components = np.arange(n_components_start, n_components_end, n_components_step)
    models = [GM_model(n, covariance_type='full', random_state=random_state).fit(X) for n in n_components]
    if is_aic:
        score_aic = [m.aic(X) for m in models]
        best_model_index = np.argmin(score_aic)
    else:
        score_bic = [m.bic(X) for m in models]
        best_model_index = np.argmin(score_bic)

    # 3. 扩充数据
    gmm = models[best_model_index]
    gmm.fit(X)
    VX = gmm.sample(expand_number)[0]

    # 4.保存数据
    fmt = "%.0{}f".format(result_retain)
    if target_name is None:
        np.savetxt("result.csv", VX, delimiter=',', fmt=fmt)
        VX_2D = pca1.transform(VX)
        plt.scatter(VX_2D[:, 0], VX_2D[:, 1], color='red')
        plt.xlabel('PCA-Dimension1')
        plt.ylabel('PCA-Dimension2')
        plt.savefig('vsg.jpg')
        # plt.show()
        data_frame = DataFrame(VX[0:10], columns=col_names)
        return data_frame
    else:
        s_scale = StandardScaler()
        s_scale.fit(X)
        standard_X = s_scale.transform(X)
        standard_VX = s_scale.transform(VX)
        _, sort_idx = calculate_dis_of_tp_and_vp_batch(standard_VX, standard_X)
        created_y = np.array([y[i[0]] for i in sort_idx])
        created_data = np.hstack((VX, created_y.reshape(-1, 1)))
        np.savetxt("result.csv", created_data, delimiter=',', fmt=fmt)
        VX_2D = pca2.transform(created_data)
        plt.scatter(VX_2D[:, 0], VX_2D[:, 1], color='red')
        plt.xlabel('PCA-Dimension1')
        plt.ylabel('PCA-Dimension2')
        plt.savefig('vsg.jpg')
        # plt.show()
        data_frame = DataFrame(created_data[0:10], columns=col_names)
        return data_frame

def _read_parameters():
    import json

    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    parameters['feature_space_start_index'] = int(parameters['feature_space_start_index'])
    parameters['feature_space_end_index'] = int(parameters['feature_space_end_index'])
    if parameters['target_name'] == '{target_name}':
        parameters['target_name'] = None
    try:
        parameters['expand_number'] = int(parameters['expand_number'])
    except:
        parameters['expand_number'] = 200
    try:
        parameters['result_retain'] = int(parameters['result_retain'])
    except:
        parameters['result_retain'] = 4
    try:
        parameters['random_state'] = int(parameters['random_state'])
    except:
        parameters['random_state'] = None
    try:
        parameters['is_aic'] = False if int(parameters['is_aic']) == 0 else True
    except:
        parameters['is_aic'] = True
    try:
        parameters['n_components_start'] = int(parameters['n_components_start'])
    except:
        parameters['n_components_start'] = 1
    try:
        parameters['n_components_step'] = int(parameters['n_components_step'])
    except:
        parameters['n_components_step'] = 1


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


def _add_info_xml(vsg_result) -> None:
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
    Picture = output.find('Picture')

    element_picture = ET.Element('picture')
    for picture_name in {'origin.jpg','vsg.jpg'}:
        element_file, element_name, element_url = ET.Element('file'), ET.Element('name'), ET.Element('url')
        element_name.text, element_url.text = picture_name, picture_name
        element_file.append(element_name)
        element_file.append(element_url)
        element_picture.append(element_file)
    Picture.append(element_picture)
    
    Demo = output.find('Demo')
    element_catalog = ET.Element('catalog')
    for _, data in vsg_result.iterrows():
        element_temp = ET.Element('values')
        for key, value in zip(data.index, data):
            temp_key = ET.Element(str(key))
            temp_key.text = str(value)
            element_temp.append(temp_key)
        element_catalog.append(element_temp)
    Demo.append(element_catalog)
    # Save
    __indent(root)
    ET.tostring(root, method='xml')

    tree.write('result.xml', encoding='utf-8', xml_declaration=True)
    with open('result.xml', 'r') as fp:
        lines = [line for line in fp]
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/VSG.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))

if __name__ == "__main__":
    # demo
    parameters = _read_parameters()
    inputCSV = parameters['inputCSV']
    expand_number = parameters['expand_number']
    feature_space_start_idx = parameters['feature_space_start_idx']
    feature_space_end_idx = parameters['feature_space_end_idx']
    target_name = parameters['target_name']
    result_retain = parameters['result_retain']
    random_state = parameters['random_state']
    is_aic = parameters['is_aic']
    n_components_start = parameters['n_components_start']
    n_components_step = parameters['n_components_step']
    vsg_result = gmm_vsg(inputCSV, expand_number, feature_space_start_idx, 
    feature_space_end_idx, target_name=target_name, result_retain=result_retain,
    random_state=random_state, is_aic=is_aic, n_components_start=n_components_step, 
    n_components_step=n_components_step
    )
    _add_info_xml(vsg_result)