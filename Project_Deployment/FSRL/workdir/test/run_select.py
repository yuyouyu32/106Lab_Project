from env import Maze
from DDQN import DDeepQNetwork
from pylab import *
import csv
import pandas as pd  # 导入pandas包
import numpy as np
from sklearn import preprocessing
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from sklearn import *

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def run_select(name_list, env, RL,index, item = 2000):

    max_reward = -100
    max_action = []
    max_score=0

    step = 0

    x_axis_data = []
    y_axis_data = []

    for episode in range(item):

        # initial observation

        observation = env.reset()

        actions=[]

        while True:

            # RL choose action based on observation

            action = RL.choose_action(observation,observation)


            actions.append(action)

            # RL take action and get next observation and reward

            observation_, reward, done, score= env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):

                RL.learn()

            # swap observation

            observation = observation_

            # break while loop when end of this episode

            if done:
                print(episode)
                x_axis_data.append(episode)
                y_axis_data.append(score)

                if reward > max_reward:
                    max_score=score

                    max_reward = reward

                    max_action.clear()

                    max_action.append(actions)

                if reward == max_reward and actions not in max_action:

                    max_action.append(actions)

                break

            step += 1

    # end of game

    matplotlib.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1)
    max_xdata=[]
    max_ydata=[]
    for i in range(len(x_axis_data)):
        if y_axis_data[i]==max_score:
            max_xdata.append(i)
            max_ydata.append(y_axis_data[i])
    plt.plot(max_xdata, max_ydata, 'ro', color='#FF0000', alpha=0.8, linewidth=1)

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('迭代次数')
    plt.ylabel(index)

    plt.savefig('result.PNG')


    return max_score,max_action, [name_list[i] for i in max_action[0]]


def read_csv(file_path, start_index, end_index, feature_index):
    data = pd.read_csv(file_path)  # 读取csv文件

    data = np.array(data)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 0:
                name_list = rows

    X = data[:, start_index:end_index]
    Y = data[:, feature_index]

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    m, n = X.shape
    return X,Y,m,n,name_list


def main(file_path, clf_str, para = None, start_index=0, end_index = -1, target_index = -1, max_choose=1, index ='r2_score', alpha=1, target=0, item=2000, learning_rate=0.05, reward_decay=0.9,e_greedy=0.9, replace_target_iter=100, memory_size=500, batch_size=32):
    X, Y, m, n, name_list = read_csv(file_path,start_index,end_index,target_index)

    tf.reset_default_graph()
    clf = None
    if clf_str == 'svc':
        if para is None:
            clf = svm.SVC(kernel='rbf')
        else:
            clf = svm.SVC(kernel='rbf',C = para["svm_c"])
    elif clf_str == 'rf_c':
        if para is None:
            clf = ensemble.RandomForestClassifier(n_estimators=100)
        else:
            clf = ensemble.RandomForestClassifier(n_estimators=para["rf_n"])
    elif clf_str == 'svr':
        if para is None:
            clf = svm.SVR(kernel='rbf')
        else:
            clf = svm.SVR(kernel='rbf',C = para["svm_c"])
    elif clf_str == 'rf_r':
        if para is None:
            clf = ensemble.RandomForestRegressor(n_estimators=100)
        else:
            clf = ensemble.RandomForestRegressor(n_estimators=para["rf_n"])
    elif clf_str == 'xgboost':
        if para is None:
            clf = ensemble.XGBRegressor(n_estimators=100)
        else:
            clf = ensemble.XGBRegressor(n_estimators=para["xgboost_n"])
    if clf == None:
        return

    env = Maze(n,clf,target, X, Y, max_choose, alpha=alpha, index=index)

    RL = DDeepQNetwork(env.n_actions, env.n_features,

                       learning_rate=learning_rate,

                       reward_decay=reward_decay,

                       e_greedy=e_greedy,

                       replace_target_iter=replace_target_iter,

                       memory_size=memory_size, batch_size=batch_size)
    max_score, max_action, best_feature = run_select(name_list ,env, RL,index, item)

    RL.plot_cost("cost.PNG")

    print("max_reward:")
    print(max_score)

    print('max_action:')
    print(max_action)

    print(best_feature)
    return max_score, max_action, best_feature





if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    import os

    os.environ['KMP_WARNINGS'] = '0'

    old_v = tf.compat.v1.logging.get_verbosity()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    clf = svm.SVC(kernel='rbf')

    target = 0.73
    clf_str = 'svc'
    main('PN_label.csv', 'svc', start_index=0, end_index=-1, target_index=-1, max_choose=10, index ='accuracy_score', target=target, item=500)















