# 导入需要的包
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# model
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler


# places to store optimal models and scores
global opt_models
opt_models = dict()

# no. k-fold splits
splits = 10
# no. k-fold iterations
repeats = 5
rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

def data_process(path,target,x_start,x_end):
    '''
    进行数据预处理
    :param path: 需要读入的文件的位置，str
    :param target: 需要预测的目标,str
    :return:
    '''
    data = pd.read_csv(path)
    data = data[data[target].notna()]
    x = data.iloc[:,x_start-1:x_end]
    y = data[target]

    print(x)
    # 预处理，正则化
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(x))

    # 画图函数，画出两个target的分布
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.distplot(y.dropna(),fit=stats.norm);
    plt.subplot(1,2,2)
    _=stats.probplot(y.dropna(), plot=plt)
    plt.show()

    return X,y


def kfold(model, X, y):
    # 5折交叉验证
    Folds = 5
    kf = KFold(n_splits=10, shuffle=True, random_state=300)
    # 记录训练和预测relavent error
    # 记录训练和预测r2
    train_r2_ = []
    test_r2_ = []

    train_re_ = []
    test_re_ = []

    # 线下训练预测
    for (train, test) in (kf.split(X, y)):
        # 切分训练集和预测集
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # 训练模型
        model.fit(X_train, y_train)

        # 训练集预测 测试集预测
        y_train_KFold_predict = model.predict(X_train)
        y_test_KFold_predict = model.predict(X_test)

        # print('第{}折 训练和预测 训练r2 预测r2'.format(i))
        # train_r2 = model.score(X_train,y_train)
        train_r2 = r2_score(y_train, y_train_KFold_predict)
        # print('------', '训练r2', train_r2, '------')
        # test_r2 = model.score(X_test,y_test)
        test_r2 = r2_score(y_test, y_test_KFold_predict)
        # print('------', '预测r2', test_r2, '------')
        train_re = mean_relative_error(y_train, y_train_KFold_predict)
        test_re = mean_relative_error(y_test, y_test_KFold_predict)

        train_r2_.append(train_r2)
        test_r2_.append(test_r2)

        train_re_.append(train_re)
        test_re_.append(test_re)

    mean_r2_train = np.mean(train_r2_)
    mean_r2_test = np.mean(test_r2_)

    mean_re_train = np.mean(train_re_)
    mean_re_test = np.mean(test_re_)

    print('------', 'train_r2', '%.4f' % mean_r2_train, '------', 'test_r2', '%.4f' % mean_r2_test, '------')

    print('------', 'train_er', '%.4f' % (1 - mean_re_train), '------', 'test_er', '%.4f' %(1 - mean_re_test), '------')

    return (mean_re_train, mean_re_test)


def mean_relative_error(y_true, y_pred, ):
    relative_error = np.average(abs(y_true - y_pred) / y_true, axis=0)
    return relative_error

def model_Ridge(X,y,param):
    model = "Ridge"

    opt_models[model] = Ridge()
    param_grid = param
    # param_grid = {'alpha': alph_range}

    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid,  cv = rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True)

    gsearch.fit(X,y)
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_Lasso(X,y,param):
    model = 'Lasso'

    opt_models[model] = Lasso()
    param_grid = param
    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv = rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True)

    gsearch.fit(X,y)
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_elasticNet(X,y,param):
    model ='ElasticNet'

    opt_models[model] = ElasticNet()
    param_grid = param
    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_SVR(X,y,param):
    model = "SVR"
    opt_models[model] = SVR()
    param_grid = param

    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_KNeighbors(X,y,param):
    model = 'KNeighbors'
    opt_models[model] = KNeighborsRegressor()
    param_grid = param
    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_GB(X,y,param):
    model = 'GradientBoosting'
    opt_models[model] = GradientBoostingRegressor()
    param_grid = param
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True,n_jobs=8)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_rf(X,y,param):
    model = 'RandomForest'
    opt_models[model] = RandomForestRegressor()

    param_grid = param

    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True,n_jobs=8)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score

def model_xgb(X,y,param):
    model = 'xgboost'

    opt_models[model] = XGBRegressor()
    param_grid = param
    # setup grid search parameters
    gsearch = GridSearchCV(opt_models[model], param_grid=param_grid, cv=rkfold,
                           # scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True,n_jobs=8)

    # search the grid
    gsearch.fit(X,y)

    # 保存最优模型
    train_score, test_score = kfold(gsearch.best_estimator_,X,y)
    opt_models[model] = gsearch.best_estimator_,gsearch.best_params_,test_score


# if __name__ == '__main__':
def main():
    X,y = data_process(
        path='test.csv',
        x_start=2,
        x_end=3,
        target="sinx_cosx"
    )

    # model_Ridge(X,y,param={'alpha': np.arange(0.25,6,0.25)})
    # #
    # model_Lasso(X,y,param={'alpha': np.arange(1e-4,1e-3,4e-5)})

    model_elasticNet(X,y,param={'alpha': np.arange(1e-4,1e-3,1e-4),
                  'l1_ratio': np.arange(0.1,1.0,0.1),
                  'max_iter':[100000]
                                })

    # model_KNeighbors(X,y,param={'n_neighbors':np.arange(1,11,1)})

    # model_SVR(X,y,param={'C':np.arange(1,20,1)
    #                      })

    # model_GB(X,y,param={'n_estimators':np.arange(150,351,100),
    #               'max_depth':np.arange(1,4,1),
    #               'min_samples_split':np.arange(5,8,1)})

    model_rf(X,y,param={'n_estimators':np.arange(100,251,50),
                  'max_features':np.arange(8,21,4),
                  'min_samples_split':np.arange(2,7,2)})

    # model_xgb(X,y,param={'n_estimators':np.arange(1,10,20),  # 分类树数量
    #                      "max_depth": np.arange(3, 4, 2),  # 每颗树的搜索深度
    #                      "eta":[0.3]  # 学习率,写死
    #                      }
    #                       )

    opt_models = pd.DataFrame(opt_models)
    print(opt_models)
    #opt_models.to_csv('result.csv')