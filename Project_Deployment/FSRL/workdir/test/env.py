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
