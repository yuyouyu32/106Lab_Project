# -*-coding=utf-8-*-
import sys

import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture as GM_model
from sklearn.preprocessing import StandardScaler


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
    if not os.path.exists(file_path):
        print("文件路径不正确，请重新输入！")
        sys.exit(-1)
    df_data = pd.read_csv(file_path, delimiter=',')
    if target_name is None:
        X = df_data.values
    else:
        X = df_data.values[:, feature_space_start_idx - 1:feature_space_end_idx - 1]
        y = df_data[target_name].values

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
    else:
        s_scale = StandardScaler()
        s_scale.fit(X)
        standard_X = s_scale.transform(X)
        standard_VX = s_scale.transform(VX)
        _, sort_idx = calculate_dis_of_tp_and_vp_batch(standard_VX, standard_X)
        created_y = np.array([y[i[0]] for i in sort_idx])
        created_data = np.hstack((VX, created_y.reshape(-1, 1)))
        np.savetxt("result.csv", created_data, delimiter=',', fmt=fmt)


if __name__ == "__main__":
    # demo
    gmm_vsg('../data/23items2.csv', 100, 2, 7, 'akronAbrasion')
