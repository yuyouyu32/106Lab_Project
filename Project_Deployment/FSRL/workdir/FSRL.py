# from env import Maze
# from DDQN import DDeepQNetwork
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

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pandas as pd  # 导入pandas包
import numpy as np
from sklearn import svm
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import *
# import tensorflow.compat.v1 as tf
from utils import _add_error_xml, _add_info_xml, _read_parameters

# np.random.seed(1)

# tf.set_random_seed(1)
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['KMP_WARNINGS'] = '0'

old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

kfold = 10


class Maze:
    def __init__(self, n_f, clf, target, X, Y,MAX_choose, alpha=1, index='r2_score'):
        self.action_space = []

        for i in range(n_f):
            self.action_space.append(i)  # action_space=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        self.n_actions = len(self.action_space)  # n_actions=14

        self.n_features = n_f  # n_features=14

        self.s_f = np.zeros(n_f, dtype=int)  # s_f=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]  (14*1)

        self.score = 0

        self.clf = clf

        self.target = target

        self.alpha = alpha

        self.X = X

        self.Y = Y

        if index == 'accuracy_score':
            self.index = accuracy_score
        else:
            self.index = r2_score
        self.max_choose = MAX_choose

    def reset(self):

        self.s_f = np.zeros(self.n_features, dtype=int)

        self.score = 0

        return self.s_f

    def step(self, action):

        done = False

        if self.s_f[action] == 0:
            self.s_f[action] = 1

        F = []

        m = 0

        for i in range(self.n_features):

            if self.s_f[i] == 1:
                F.append(i)

                m += 1

        # reward function

        score = 0

        if m == self.max_choose:
            skf = KFold(n_splits=kfold, shuffle=True, random_state=1)  # 各个类别的比例大致和完整数据集中相同

            clf = self.clf

            x = self.X

            y = self.Y

            for train_index, test_index in skf.split(x, y):

                x_ktrain, x_ktest = x[train_index], x[test_index]

                y_ktrain, y_ktest = y[train_index], y[test_index]

                clf.fit(x_ktrain[:, F], y_ktrain)

                y_predict = clf.predict(x_ktest[:, F])

                score += self.index(y_ktest, y_predict)

            score = score / kfold

            rewards=(score - self.target) * self.alpha
            done = True

        else:

            rewards = 0

            done = False

        s_ = self.s_f

        return s_, rewards, done, score


class DDeepQNetwork:

    def __init__(

            self,

            n_actions,

            n_features,

            learning_rate=0.01,

            reward_decay=0.9,

            e_greedy=0.9,

            replace_target_iter=500,

            memory_size=2000,

            batch_size=400,

            e_greedy_increment=None,

            output_graph=False,

            # prioritized=True,

    ):
        import os
        os.environ['KMP_WARNINGS'] = '0'

        self.n_actions = n_actions

        self.n_features = n_features

        self.lr = learning_rate

        self.gamma = reward_decay

        self.epsilon_max = e_greedy

        self.replace_target_iter = replace_target_iter

        self.memory_size = memory_size

        self.batch_size = batch_size

        self.epsilon_increment = e_greedy_increment

        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # self.prioritized = prioritized  # decide to use double q or not

        # total learning step

        self.learn_step_counter = 0

        # # initialize zero memory [s, a, r, s_]
        #
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):

            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:

            # $ tensorboard --logdir=logs

            tf.summary.FileWriter("logs/", self.sess.graph)



        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []



    def _build_net(self):

        # ------------------ all inputs ------------------------

        # tf.compat.v1.disable_eager_execution()


        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State

        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward

        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # tf.set_random_seed(1)

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)




        # ------------------ build evaluate_net ------------------

        with tf.variable_scope('eval_net'):

            e1 = tf.layers.dense(self.s, 1000, tf.nn.relu, kernel_initializer=w_initializer,

                                 bias_initializer=b_initializer, name='e1')

            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,

                                          bias_initializer=b_initializer, name='q')



        # ------------------ build target_net ------------------

        with tf.variable_scope('target_net'):

            t1 = tf.layers.dense(self.s_, 1000, tf.nn.relu, kernel_initializer=w_initializer,

                                 bias_initializer=b_initializer, name='t1')

            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,

                                          bias_initializer=b_initializer, name='t2')



        with tf.variable_scope('q_target'):

            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )

            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)

            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):

            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)



    def store_transition(self, s, a, r, s_):

        if not hasattr(self, 'memory_counter'):

            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory

        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition

        self.memory_counter += 1


    def choose_action(self, observation,s_f):

        # to have batch dimension when feed into tf placeholder

        observation = observation[np.newaxis, :]
        # print(self.sess.run(self.q_eval, feed_dict={self.s: observation}))

        if np.random.uniform() < self.epsilon:

            # forward feed the observation and get q value for every actions

            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

            actions = np.argmax(actions_value)

            while (s_f[actions] == 1):

                actions_value[0][actions] = actions_value[0][np.argmin(actions_value)] - 0.1

                actions = np.argmax(actions_value)

        else:

            actions = np.random.randint(0, self.n_actions)

            while (s_f[actions] == 1):

                actions = np.random.randint(0, self.n_actions)

        return actions

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

            # print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, cost = self.sess.run(

            [self._train_op, self.loss],

            feed_dict={

                self.s: batch_memory[:, :self.n_features],

                self.a: batch_memory[:, self.n_features],

                self.r: batch_memory[:, self.n_features + 1],

                self.s_: batch_memory[:, -self.n_features:],

            })

        self.cost_his.append(cost)

        # increasing epsilon

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1




    def plot_cost(self, name):

        import matplotlib.pyplot as plt
        plt.clf()

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)

        plt.ylabel('Cost')

        plt.xlabel('training steps')

        # plt.show()
        plt.savefig(name)

    def choose_best_action(self, observation,s_f):

        # to have batch dimension when feed into tf placeholder

        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

        actions = np.argmax(actions_value)

        while (s_f[actions] == 1):

            actions_value[0][actions] = actions_value[0][np.argmin(actions_value)] - 0.1

            actions = np.argmax(actions_value)

        return actions

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
    plt.xlabel('epochs')
    plt.ylabel(index)

    plt.savefig('result.png')


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


def FSRL(file_path, clf_str, para = None, start_index=0, end_index = -1, target_index = -1, max_choose=1, index ='r2_score', alpha=1, target=0, item=2000, learning_rate=0.05, reward_decay=0.9,e_greedy=0.9, replace_target_iter=100, memory_size=500, batch_size=32):
    X, Y, m, n, name_list = read_csv(file_path,start_index,end_index,target_index)

    tf.reset_default_graph()
    clf = None
    if clf_str == 'svc':
        if para is None:
            clf = svm.SVC(kernel='rbf')
        else:
            clf = svm.SVC(kernel='rbf',C = para)
    elif clf_str == 'rf_c':
        if para is None:
            clf = ensemble.RandomForestClassifier(n_estimators=100)
        else:
            clf = ensemble.RandomForestClassifier(n_estimators=para)
    elif clf_str == 'svr':
        if para is None:
            clf = svm.SVR(kernel='rbf')
        else:
            clf = svm.SVR(kernel='rbf',C = para)
    elif clf_str == 'rf_r':
        if para is None:
            clf = ensemble.RandomForestRegressor(n_estimators=100)
        else:
            clf = ensemble.RandomForestRegressor(n_estimators=para)
    elif clf_str == 'xgboost':
        if para is None:
            clf = XGBRegressor(n_estimators=100)
        else:
            clf = XGBRegressor(n_estimators=para)
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

    RL.plot_cost("cost.png")

    print("max_reward:")
    print(max_score)

    print('max_action:')
    print(max_action)
    
    print(best_feature)
    return max_score, max_action, best_feature

def main():
    try:
        parameters = _read_parameters()
        inputCSV = parameters["inputCSV"]
        start_index = parameters["start_index"]
        end_index = parameters["end_index"]
        target_index = parameters["target_index"]
        max_choose=parameters["max_choose"]
        target_score = parameters["target_score"]
        clf_str = parameters['model']
        index=parameters['score_kind']
        para = parameters['parameter']
        item=parameters['epochs']
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        max_score, max_action, best_feature = FSRL(file_path=inputCSV, 
                                                    clf_str=clf_str, para=para, start_index=start_index-1, end_index=end_index, target_index=target_index-1, max_choose=max_choose,
                                                    index=index, alpha=1, target=target_score, item=item, 
                                                    learning_rate=0.05, reward_decay=0.9,e_greedy=0.9, replace_target_iter=100, memory_size=500, batch_size=32)
    except Exception as e:
        _add_error_xml("FSRL Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return 
    
    try:
        _add_info_xml(picture_names=['result.png', 'cost.png'], names=best_feature, indexs=max_action, score=max_score)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
    
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')



if __name__ == "__main__":
    import sys
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close















