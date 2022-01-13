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
import sys

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

    # print(x)
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
    plt.savefig('Distribution_Chart.png')

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
                           verbose=1, return_train_score=True)

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
                           verbose=1, return_train_score=True)

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


def _read_parameters():
    import json
    def re_input(original: str):
        return original.replace('，', ',').replace(' ', '')
    def split2int(original: str):
        return list(map(int, original.split(',')))

    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    # Data
    parameters['feature_start'] = int(parameters['feature_start'])
    parameters['feature_end'] = int(parameters['feature_end'])
    parameters['target_name'] = parameters['target_name']

    # Ridge
    parameters['Ridge_alpha'] = np.arange(float(parameters['Ridge_alpha_start']),
                                            float(parameters['Ridge_alpha_end']), 
                                            float(parameters['Ridge_alpha_step']))

    # Lasso
    parameters['Lasso_alpha'] = np.arange(float(parameters['Lasso_alpha_start']), 
                                            float(parameters['Lasso_alpha_end']),
                                            float(parameters['Lasso_alpha_step']))

    # elasticNet
    parameters['elasticNet_alpha'] = np.arange(float(parameters['elasticNet_alpha_start']),
                                                float(parameters['elasticNet_alpha_end']),
                                                float(parameters['elasticNet_alpha_step']))
    parameters['elasticNet_l1_ratio'] = np.arange(float(parameters['elasticNet_l1_ratio_start']),
                                                float(parameters['elasticNet_l1_ratio_end']),
                                                float(parameters['elasticNet_l1_ratio_step']))
    parameters['elasticNet_max_iter'] = np.arange(int(parameters['elasticNet_max_iter_start']),
                                                int(parameters['elasticNet_max_iter_end']),
                                                int(parameters['elasticNet_max_iter_step']))    

    # Kneighbors
    parameters['Kneighbors_n_neighbors'] = np.arange(int(parameters['Kneighbors_n_neighbors_start']),
                                                int(parameters['Kneighbors_n_neighbors_end']),
                                                int(parameters['Kneighbors_n_neighbors_step']))

    # SVR                                         
    parameters['SVR_C'] = np.arange(int(parameters['SVR_C_start']),
                                                int(parameters['SVR_C_end']),
                                                int(parameters['SVR_C_step']))

    # GB
    parameters['GB_n_estimators'] = np.arange(int(parameters['GB_n_estimators_start']),
                                                int(parameters['GB_n_estimators_end']),
                                                int(parameters['GB_n_estimators_step']))
    
    parameters['GB_max_depth'] = np.arange(int(parameters['GB_max_depth_start']),
                                                int(parameters['GB_max_depth_end']),
                                                int(parameters['GB_max_depth_step']))
    parameters['GB_min_samples_split'] = np.arange(int(parameters['GB_min_samples_split_start']),
                                                int(parameters['GB_min_samples_split_end']),
                                                int(parameters['GB_min_samples_split_step']))


    # RF                                          
    parameters['RF_n_estimators'] = np.arange(int(parameters['RF_n_estimators_start']),
                                                int(parameters['RF_n_estimators_end']),
                                                int(parameters['RF_n_estimators_step']))
    
    parameters['RF_max_features'] = np.arange(float(parameters['RF_max_features_start']),
                                                float(parameters['RF_max_features_end']),
                                                float(parameters['RF_max_features_step']))
    parameters['RF_min_samples_split'] = np.arange(int(parameters['RF_min_samples_split_start']),
                                                int(parameters['RF_min_samples_split_end']),
                                                int(parameters['RF_min_samples_split_step']))

    # XGBoost
    parameters['xgb_n_estimators'] = np.arange(int(parameters['xgb_n_estimators_start']),
                                                int(parameters['xgb_n_estimators_end']),
                                                int(parameters['xgb_n_estimators_step']))
    
    parameters['xgb_max_depth'] = np.arange(int(parameters['xgb_max_depth_start']),
                                                int(parameters['xgb_max_depth_end']),
                                                int(parameters['xgb_max_depth_step']))
    parameters['xgb_eta'] = np.arange(float(parameters['xgb_eta_start']),
                                                float(parameters['xgb_eta_end']),
                                                float(parameters['xgb_eta_step']))                                         
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


def _add_info_xml(picture_name, result_models) -> None:
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

    # Models
    for module_name, module_para in result_models.items():
        element_module = ET.Element(module_name)
        for para_name, para_value in module_para[1].items():
            element_para = ET.Element(para_name)
            try:
                element_para.text = str(round(para_value, 6))
            except:
                element_para.text = str(para_value)
            element_module.append(element_para)
        element_R2 = ET.Element('R2')
        element_R2.text = str(round(module_para[2], 6))
        element_module.append(element_R2)
        output.append(element_module)
    # Picture
    element_pictures = ET.Element('Distribution_Chart')
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
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/Hyper_Optimize.xsl" ?>\n')
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
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/Hyper_Optimize.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))

def main():
    global opt_models
    try:
        parameters = _read_parameters()
        X,y = data_process(
        path=parameters['inputCSV'],
        x_start=parameters['feature_start'],
        x_end=parameters['feature_end'],
        target=parameters['target_name']
        )
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    Error_num = 0
    try:
        print('Ridge...')
        model_Ridge(X,y,param={'alpha': parameters['Ridge_alpha']})
    except Exception as e:
        Error_num += 1
        print('Ridge Error.')

    try:
        print('Lasso...')
        model_Lasso(X,y,param={'alpha': parameters['Lasso_alpha']})
    except Exception as e:
        Error_num += 1
        print('Lasso Error.')

    try:
        print('elasticNet...')
        model_elasticNet(X,y,param={'alpha': parameters['elasticNet_alpha'],
                    'l1_ratio': parameters['elasticNet_l1_ratio'],
                    'max_iter': parameters['elasticNet_max_iter']})
    except Exception as e:
        Error_num += 1
        print('elasticNet Error.')   

    try: 
        print('Kneighbors...')
        model_KNeighbors(X,y,param={'n_neighbors':parameters['Kneighbors_n_neighbors']})
    except Exception as e:
        Error_num += 1
        print('Kneighbors Error.') 
    try:
        print('SVR...')
        model_SVR(X,y,param={'C':parameters['SVR_C'],
                            'kernel':('linear', 'rbf')
                            })
    except Exception as e:
        Error_num += 1
        print('SVR Error.')   

    try:
        print('GB...')
        model_GB(X,y,param={'n_estimators':parameters['GB_n_estimators'],
                    'max_depth':parameters['GB_max_depth'],
                    'min_samples_split':parameters['GB_min_samples_split']})
    except Exception as e:
        Error_num += 1
        print('GB Error.')  

    try: 
        print('RF...')
        model_rf(X,y,param={'n_estimators':parameters['RF_n_estimators'],
                    'max_features':parameters['RF_max_features'],
                    'min_samples_split':parameters['RF_min_samples_split']})
    except Exception as e:
        Error_num += 1
        print('RF Error.')    

    try:             
        print('XGBoost...')
        model_xgb(X,y,param={'n_estimators':parameters['xgb_n_estimators'],  # 分类树数量
                            "max_depth": parameters['xgb_max_depth'],  # 每颗树的搜索深度
                            "eta":parameters['xgb_eta']  # 学习率,写死
                            }
                            )
    except Exception as e:
        Error_num += 1
        print('XGBoost Error.')

    if Error_num == 8:
        _add_error_xml("Hyper Optimize Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        _add_info_xml('Distribution_Chart.png', opt_models)
        opt_models_csv = pd.DataFrame(opt_models)
        opt_models_csv.to_csv('result.csv')
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return    
    
    # 'Distribution_Chart.png' 'result.csv'
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')
    return opt_models
    

if __name__ == '__main__':
    console = sys.stdout
    file = open("task_log.txt", "w")
    sys.stdout = file
    main()
    sys.stdout = console
    file.close