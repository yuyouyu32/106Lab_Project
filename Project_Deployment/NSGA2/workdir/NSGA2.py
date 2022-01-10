import numpy as np
import geatpy as ea
import csv
from sklearn import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import warnings
from utils import _add_error_xml, _add_info_xml, _read_parameters
warnings.filterwarnings("ignore")
from xgboost.sklearn import XGBRegressor

# -*- coding: utf-8 -*-
from sys import path as paths
from os import path
import sys

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])

# 修改拥挤距离排序的版本
class moea_NSGA2_preference_templet(ea.MoeaAlgorithm):
    """
    moea_NSGA2_templet : class - 多目标进化NSGA-II算法模板

    算法描述:
        采用NSGA-II进行多目标优化，算法详见参考文献[1]。

    模板使用注意:
        本模板调用的目标函数形如：aimFunc(pop),
        其中pop为Population类的对象，代表一个种群，
        pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
        该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
        若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
        该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中，
                            违反约束程度矩阵保存在种群对象的CV属性中。
        例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
            此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
        若不符合上述规范，则请修改算法模板或自定义新算法模板。

    参考文献:
        [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective
        genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary
        Computation, 2002, 6(2):0-197.

    """

    def __init__(self, problem, population, target=None, weight=None):
        self.target = target
        self.weight = weight
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=1)  # 生成二进制变异算子对象
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def reinsertion(self, population, offspring, NUM):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。
        """

        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NUM, None,
                                         population.CV)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离

        if self.target is not None:
            count = 0
            cos_arr = np.zeros(dis.shape)
            # 计算余弦相似度
            for i in population.ObjV:
                res = np.array(
                    [[i[j] * self.target[j], i[j] * i[j], self.target[j] * self.target[j]] for j in range(len(i))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
                cos_arr[count] += cos
                count += 1
            cos_arr = cos_arr * self.weight
            dis = cos_arr + dis

        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def run(self):
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        if population.Chrom is None:
            population.initChrom()  # 初始化种群染色体矩阵（内含解码，详见Population类的源码）
        else:
            population.Phen = population.decoding()  # 染色体解码
        self.problem.aimFunc(population)  # 计算种群的目标函数值
        self.evalsNum = population.sizes  # 记录评价次数
        [levels, criLevel] = self.ndSort(self.problem.maxormins * population.ObjV, NIND, None,
                                         population.CV)  # 对NIND个个体进行非支配分层
        population.FitnV[:, 0] = 1 / levels  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            offspring.Phen = offspring.decoding()  # 解码
            self.problem.aimFunc(offspring)  # 求进化后个体的目标函数值
            self.evalsNum += offspring.sizes  # 更新评价次数
            # 重插入生成新一代种群
            population = self.reinsertion(population, offspring, NIND)

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果


class MyProblem(ea.Problem):
    def __init__(self, tatget_number, optimization_direction, dimension, feature_types, feature_lower, feature_upper, is_lower, is_upper, model_list):
        name='problem'
        # 优化目标的维度,类型为number
        M=tatget_number
        # 目标最大最小标记列表，类型为一维列表，数组长度为tatget_number，1表示最大化该目标，-1表示最小化该目标
        maxormins = optimization_direction
        # 自变量个数
        Dim = dimension
        # 自变量类型范围，1表示整数，0表示实数，类型为列表，长度等于dimension
        varTypes = feature_types
        # 自变量下界，类型为列表，长度等于dimension
        lb = feature_lower
        # 自变量上界，类型为列表，长度等于dimension
        ub = feature_upper
        # 自变量能否等于下界，类型为列表，长度等于dimension，0表示不含边界值，1表示包含边界值
        lbin = is_lower
        # 自变量能否等于上界，类型为列表，长度等于dimension
        ubin = is_upper
        self.model_list = model_list
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
    def aimFunc(self, pop):
        Vars=pop.Phen
        result = []
        for model in self.model_list:
            result.append(model.predict(Vars))
        pop.ObjV = np.array(result).T

# 参数依次为model_list 训练好的两个模型，在main()里通过训练csv里数据生成
# tatget_number要优化的目标，optimization_direction优化方向[1表示最大化，0表示最小化]，
# dimension自变量个数，feature_types自变量的类型[1表示整数，0表示实数]，
# feature_lower自变量下界，feature_upper自变量上界
# is_lower，is_upper能否取得上下界0表示不含，1表示包含
# population_number表示种群个数，max_ge最大迭代次数
# cross_possibility交叉可能性
# preference 偏好方向(可不填), weight偏好权重(偏好方向不填，这个就无意义)
def nsga2(model_list, tatget_number, optimization_direction, dimension, feature_types, feature_lower, feature_upper, is_lower, is_upper,population_number, max_ge, cross_possibility = 0.8, preference=None, weight=None):
    problem = MyProblem(tatget_number, optimization_direction, dimension, feature_types, feature_lower, feature_upper, is_lower, is_upper,model_list)
    Encoding = 'RI'
    NIND = population_number
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    # preference 为偏好，默认为空，输入为长度为optimization_direction的列表，表示偏好所占权值，数字类型，范围是0~无穷大
    myAl = moea_NSGA2_preference_templet(problem, population, target=preference, weight = weight)
    myAl.MAXGEN = max_ge
    myAl.recOper.XOVR = cross_possibility
    myAl.drawing = 0
    NDSet = myAl.run()
    NDSet.save()

    if tatget_number <= 2:
        save(NDSet.ObjV)
    # 将Result中的两个文件取出
    alllist = os.listdir()
    if 'Result' in alllist:
        shutil.copyfile('./Result/ObjV.csv', 'ObjV.csv')
        shutil.copyfile('./Result/Phen.csv', 'Phen.csv')
    return NDSet.Phen, NDSet.ObjV

def save(ObjV):
    x = [i[0] for i in ObjV]
    y = [i[1] for i in ObjV]

    plt.scatter(x, y, c='red', s=10)
    plt.savefig('result.png')




def str2model(clf_str,para):
    clf = None
    if clf_str == 'svr':
        if para is None:
            clf = svm.SVR(kernel='rbf')
        else:
            clf = svm.SVR(kernel='rbf', C=para)
    elif clf_str == 'rf':
        if para is None:
            clf = ensemble.RandomForestRegressor(n_estimators=100)
        else:
            clf = ensemble.RandomForestRegressor(n_estimators=para)
    elif clf_str == 'xgboost':
        if para is None:
            clf = XGBRegressor(n_estimators=100)
        else:
            clf = XGBRegressor(n_estimators=para)
    else:
        clf = svm.SVR(kernel='rbf')
    return clf

def read_csv(file_path, start_index, end_index, target_index):
    data = pd.read_csv(file_path)  # 读取csv文件

    data = np.array(data)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 0:
                name_list = rows

    X = data[:, start_index:end_index]
    Y = []
    for i in target_index:
        Y.append(data[:, i])
    return X,Y,name_list


def preWithNSGA(file_name, target_number,clf_str,para,start_index,end_index,target_index):
    X, Y, name_list = read_csv(file_name, start_index, end_index, target_index)
    model_list = []
    for i in range(target_number):
        clf = str2model(clf_str[i],para[i])
        model_list.append(clf.fit(X,Y[i]))
    return model_list,name_list

def arr2frame(name_list, start_index,end_index,target_index, Phen, ObjV):
    feature_name = name_list[start_index:end_index]
    target_name = [name_list[i] for i in target_index]
    feature_name.extend(target_name)
    result = []
    for i in range(len(Phen)):
        result.append(np.append(Phen[i],ObjV[i]))
    result = np.array(result)
    frame = pd.DataFrame(result)
    frame.columns = feature_name

    return frame

def NSGA2(file_name, target_number,
         clf_str,para,start_index,
         end_index,target_index,tatget_number,
         optimization_direction, dimension, feature_types,
         feature_lower, feature_upper, is_lower, is_upper,population_number,
         max_ge, cross_possibility = 0.8, preference=None, weight=None):
    model_list, name_list = preWithNSGA(file_name, target_number, clf_str, para, start_index, end_index, target_index)

    Phen, ObjV = nsga2(model_list, tatget_number,
         optimization_direction, dimension, feature_types,
         feature_lower, feature_upper, is_lower, is_upper,population_number,
         max_ge, cross_possibility = cross_possibility, preference=preference, weight=weight)


    frame = arr2frame(name_list, start_index, end_index, target_index, Phen, ObjV)
    return Phen, ObjV, frame



def main():
    try:
        parameters = _read_parameters()
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        Phen, ObjV, frame = NSGA2(file_name=parameters["inputCSV"], target_number=parameters['target_number'], clf_str=parameters['models'], para=parameters['parameter'],
                             start_index=parameters['start_index']-1, end_index=parameters['end_index'], target_index=map(lambda x:x-1, parameters['target_index']), 
                             tatget_number=parameters['target_number'], optimization_direction=parameters['optimization_direction'],
                             dimension=parameters['dimension'], feature_types=parameters['feature_types'], feature_lower=parameters['feature_lower'], feature_upper=parameters['feature_upper'],
                             is_lower=parameters['is_lower'], is_upper=parameters['is_upper'], population_number=parameters['population_number'], 
                             max_ge=parameters['max_ge'], cross_possibility=parameters['cross_possibility'],
                             preference=parameters['preference'],weight=parameters['weight'])
    except Exception as e:
        _add_error_xml("NSGA2 Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return 
    try:
        if parameters['target_number'] == 2:
            picture_names = ['result.png']
        else:
            picture_names = None
        _add_info_xml(picture_names, result=frame)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
    
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')

if __name__=='__main__':
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close

    



