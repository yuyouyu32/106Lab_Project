import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
import shap
from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")


def _read_parameters():
    import json
    def re_input(original: str):
        return original.replace('，', ',').replace(' ', '')
    def split2int(original: str):
        return list(map(int, original.split(',')))

    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    parameters['feature_start'] = int(parameters['feature_start'])
    parameters['feature_end'] = int(parameters['feature_end'])
    # SHAP
    try:
        temp = split2int(re_input(parameters['single']))
        parameters['single'] = temp - 1
    except:
        parameters['single'] = None
    try:
        parameters['interactionA'] = split2int(re_input(parameters['interactionA']))
        parameters['interactionB'] = split2int(re_input(parameters['interactionB']))
        parameters['interaction'] = [[A - 1, B - 1] for A,B in zip(parameters['interactionA'], parameters['interactionB'])]
    except:
        parameters['interaction'] = None

    # feature selection
    try:
        parameters['Pearson'] = int(parameters['Pearson'])
    except:
        parameters['Pearson'] = None
    try:
        parameters['Variance'] = int(parameters['Variance'])
    except:
        parameters['Variance'] = None
    try:
        parameters['Lasso'] = int(parameters['Lasso'])
    except:
        parameters['Lasso'] = None
    try:
        parameters['ElasticNet'] = int(parameters['ElasticNet'])
    except:
        parameters['ElasticNet'] = None
    try:
        parameters['SVR'] = int(parameters['SVR'])
    except:
        parameters['SVR'] = None
    try:
        parameters['SVR_RFE'] = int(parameters['SVR_RFE'])
    except:
        parameters['SVR_RFE'] = None
    try:
        parameters['RF'] = int(parameters['RF'])
    except:
        parameters['RF'] = None
    try:
        parameters['LightGBM'] = int(parameters['LightGBM'])
    except:
        parameters['LightGBM'] = None
    try:
        parameters['XGBoost'] = int(parameters['XGBoost'])
    except:
        parameters['XGBoost'] = None
    try:
        parameters['MIC'] = int(parameters['MIC'])
    except:
        parameters['MIC'] = None

    return parameters


def SHAP(inputCSV, feature_start, feature_end, target, single=None, interaction=None):
    data = pd.read_csv(inputCSV)
    cols = data.columns
    features = []
    for i in range(feature_start-1, feature_end):
        features.append(cols[i])
    # print(features)

    model = xgb.XGBRegressor(max_depth=20, learning_rate=0.05, n_estimators=150)
    model.fit(data[features], data[target].values)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(data[features], check_additivity=False)

    shap.initjs()

    picture_names = {}

    summary_names = []
    shap.summary_plot(shap_values, data[features], show=False)
    name = 'summary_shap_values.png'
    summary_names.append(name)
    plt.savefig('./'+name, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, data[features], plot_type="bar", show=False)
    name = 'summary_importance_values.png'
    summary_names.append(name)
    plt.savefig('./'+name, bbox_inches='tight')
    plt.close()

    picture_names['summary_values'] = summary_names


    if single:
        single_names = []
        for i in single:
            shap.dependence_plot(features[i], shap_values, data[features], interaction_index=None, show=False)
            name = 'shap_values_'+features[i]+'.png'
            single_names.append(name)
            plt.savefig('./'+name, bbox_inches='tight')
            plt.close()
        picture_names['single_feature'] = single_names


    #shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[features])
    if interaction:
        interaction_names = []
        for interaction_A, interaction_B in interaction:
            shap.dependence_plot(features[interaction_A], shap_values, data[features], interaction_index=features[interaction_B], show=False)
            name = 'interaction_values_'+features[interaction_A]+'_inter_'+features[interaction_B]+'.png'
            interaction_names.append(name)
            plt.savefig('./'+name, bbox_inches='tight')
            plt.close()
        picture_names['interaction_feature'] = interaction_names

    return picture_names


def feature_selection_regression(X,y,feature_name, num_feats = 10, Pearson=None, Variance=None, MIC=None, Lasso=None, ElasticNet=None, SVR=None,\
                                SVR_RFE=None, RF=None, LightGBM=None, XGBoost=None): 
    weight = {'Pearson': 2,
        'Variance': 1,
        'MIC': 2,
        'Lasso': 2,
        'ElasticNet':2,
        'SVR':3,
        'SVR_RFE':6,
        'RF':7,
        'LightGBM':6,
        'XGBoost':8
        }
    if(Pearson):
        weight['Pearson']=Pearson
    if(Variance):
        weight['Variance']=Variance
    if(MIC):
        weight['MIC']=MIC
    if(Lasso):
        weight['Lasso']=Lasso
    if(ElasticNet):
        weight['ElasticNet']=ElasticNet
    if(SVR):
        weight['SVR']=SVR
    if(SVR_RFE):
        weight['SVR_RFE']=SVR_RFE
    if(RF):
        weight['RF']=RF
    if(LightGBM):
        weight['LightGBM']=LightGBM
    if(XGBoost):
        weight['XGBoost']=XGBoost
    
    
    def Normalization(X):
        X = X / np.sum(X);
        return X*100

    min_max_scaler = preprocessing.MinMaxScaler()
    list_values = [i for i in weight.values()]
    weight_value = Normalization(list_values)
    
    #weight_value = min_max_scaler.fit_transform(np.asarray(list_values).reshape(-1,1))

    i = 0
    for key in weight.keys():
        weight[key] = weight_value[i]
        # weight[key] = weight_value[i][0]
        i = i+1

    # print (weight)

    cols = ['Pearson','Variance','MIC', 'Lasso','ElasticNet','SVR','SVR_RFE','RF', 'LightGBM', 'XGBoost']

    def df_process(data):
        for col in cols:
            data[col]=data[col].map({True:1,False:0})
        df = data[cols].apply(pd.to_numeric, downcast='float')
        df = df.mul(pd.Series(weight), axis=1)

        return pd.concat([data['Feature'],df], axis=1)

    
    
    if (num_feats > X.shape[1]):
        num_feats = X.shape[1]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    XT = scaler.transform(X)

    ###Use Pearson Correlation
    def cor_selector(X, y,num_feats):
        cor_list = []
        feature_name = X.columns.tolist()
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature

    cor_support, cor_feature = cor_selector(X, y,num_feats)
    #print(str(len(cor_feature)), 'cor:selected features')

    # 方差法
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold()
    variance_selector.fit_transform(X)
    variance_support = variance_selector.get_support()
    variance_feature = X.loc[:,variance_support].columns.tolist()
    #print(str(len(variance_feature)), 'Variance:selected features')

    # Maximal Information Coefficient(MIC)
    try:
        from minepy import MINE
        def mic_selector(X, y, num_feats):
            mic_score = []
            m = MINE()
            for column in X.columns:
                m.compute_score(X[column], y)
                mic_score.append(m.mic())
            # feature name
            mic_feature = X.iloc[:,np.argsort(np.abs(mic_score))[-num_feats:]].columns.tolist()
            mic_support = [True if i in mic_feature else False for i in feature_name]
            return mic_support, mic_feature
        mic_support, mic_feature = mic_selector(X, y,num_feats)
    except :
        mic_support = [False] * len(feature_name)
    # print(str(len(mic_feature)), 'mic:selected features')


    ### Use ElasticNet to select variables
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import ElasticNet

    # embeded_elas_selector = SelectFromModel(ElasticNet(alpha=0.01, l1_ratio=0.5), max_features=num_feats)
    # embeded_elas_selector.fit(XT, y)
    # embeded_elas_support = embeded_elas_selector.get_support()
    # embeded_elas_feature = X.loc[:, embeded_elas_support].columns.tolist()
    # print(str(len(embeded_elas_feature)), 'elasNet:selected features')

    ##ElasticNet with RFE
    embeded_elas_model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    elas_rfe = RFE(embeded_elas_model,n_features_to_select=num_feats)
    elas_rfe.fit(XT,y)
    embeded_elas_support = elas_rfe.support_
    embeded_elas_feature = X.loc[:, elas_rfe.support_].columns.tolist()
    #print(str(len(embeded_elas_feature)), 'elasNet-rfe:selected features')


    ## Use Lasso to select variables
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso

    embeded_lr_selector = SelectFromModel(Lasso(alpha = 0.01), max_features=num_feats)
    embeded_lr_selector.fit(XT, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()
    #print(str(len(embeded_lr_feature)), 'lasso:selected features')

    # ### Use SVR to select variables
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import SVR
    from sklearn.feature_selection import RFE

    embeded_svr_selector = SelectFromModel(SVR(kernel="linear"), max_features=num_feats)
    embeded_svr_selector.fit(XT, y)
    embeded_svr_support = embeded_svr_selector.get_support()
    # #print(embeded_svr_support)
    #print(str(len(embeded_svr_support)), 'svr:selected features')

    # ##SVR with RFE--计算量太大
    embeded_svr_rfe_model = SVR(kernel="linear")
    svr_rfe = RFE(embeded_svr_rfe_model,n_features_to_select=num_feats)
    svr_rfe.fit(XT,y)
    embeded_svr_rfe_support = svr_rfe.support_
    #embeded_svr_feature = X.loc[:, embeded_svr_rfe_support.support_].columns.tolist()
    # # print(str(len(embeded_svr_feature)), 'svr-rfe:selected features')

    # #
    #  use RandomForest to select features based on feature importance
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor

    embeded_rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    #print(str(len(embeded_rf_feature)), 'rf:selected features')


    ### used a LightGBM. Or an XGBoost object as long it has a feature_importances_ attribute.
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMRegressor

    lgbr=LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embeded_lgb_selector = SelectFromModel(lgbr, max_features=num_feats)
    embeded_lgb_selector.fit(X, y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    #print(str(len(embeded_lgb_feature)), 'lgb:selected features')

    ### XGBoost
    import xgboost as xgb
    xgbr = xgb.XGBRegressor(n_estimators=500,
                             learning_rate=0.1,
                             max_depth=30,
                             gamma=0.,
                             subsample=0.9,
                             colsample_bytree=0.8,
                             objective="reg:squarederror",
                             scale_pos_weight=1,
                             n_jobs=12)
    xgbr.fit(X, y)
    #feature name
    feature_list = xgbr.feature_importances_
    embeded_xgb_feature = X.iloc[:,np.argsort(np.abs(feature_list))[-num_feats:]].columns.tolist()
    #feature selection? 0 for not select, 1 for select
    embeded_xgb_support = [True if i in embeded_xgb_feature else False for i in feature_name]
    #print(str(len(embeded_xgb_feature)), 'xgb:selected features')


    # put all selection together
    # feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
    #                                     'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})

    dict = {'Feature':feature_name, 'Pearson':cor_support,'Variance':variance_support,'MIC':mic_support,'Lasso':embeded_lr_support, 'ElasticNet':embeded_elas_support,
            'SVR':embeded_svr_support, 'SVR_RFE':embeded_svr_rfe_support,'RF':embeded_rf_support,
            'LightGBM':embeded_lgb_support, 'XGBoost':embeded_xgb_support}

    # dict = {'Feature':feature_name, 'Pearson':cor_support,'Variance':variance_support,'MIC':mic_support,'Lasso':embeded_lr_support, 'ElasticNet':embeded_elas_support,
    #         'SVR':embeded_svr_support, 'SVR_RFE':embeded_svr_rfe_support,'RF':embeded_rf_support,
    #         'LightGBM':embeded_lgb_support, 'XGBoost':embeded_xgb_support}

    feature_selection_df = pd.DataFrame(dict)

    # count the selected times for each feature
    feature_selection_df = df_process(feature_selection_df)
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    
    feature_selection_df.to_csv('feature_selection.csv')

    #print("The total features number:"+str(X.shape[1]))
    #print(feature_selection_df.head(80))
    
    return feature_selection_df


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


def _add_info_xml(picture_names, feature_selection_result) -> None:
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
    SHAP = output.find('SHAP')

    
    for key, values in picture_names.items():
        element_class = ET.Element(key)
        element_picture = ET.Element('picture')
        for pic_name in values:
            element_file, element_name, element_url = ET.Element('file'), ET.Element('name'), ET.Element('url')
            element_name.text, element_url.text = pic_name, pic_name
            element_file.append(element_name)
            element_file.append(element_url)
            element_picture.append(element_file)
        element_class.append(element_picture)
        SHAP.append(element_class)
    
    feature_selection = output.find('feature-selection')
    element_catalog = ET.Element('catalog')
    for _, data in feature_selection_result.iterrows():
        feature = ET.Element('feature')
        for key, value in zip(data.index, data):
            method = ET.Element(key)
            try:
                method.text = str(round(value, 2))
            except:
                method.text = str(value)
            feature.append(method)
        element_catalog.append(feature)
    feature_selection.append(element_catalog)
    # Save
    __indent(root)
    ET.tostring(root, method='xml')

    tree.write('result.xml', encoding='utf-8', xml_declaration=True)
    with open('result.xml', 'r') as fp:
        lines = [line for line in fp]
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/feature_selection.xsl" ?>\n')
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
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/feature_selection.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))


def main():
    try:
        parameters = _read_parameters()
        inputCSV = parameters['inputCSV']
        feature_start = parameters['feature_start']
        feature_end = parameters['feature_end']
        target = parameters['target']
        # SHAP
        single = parameters['single']
        interaction = parameters['interaction']
        # feature selection
        Pearson = parameters['Pearson']
        Variance = parameters['Variance']
        Lasso = parameters['Lasso']
        ElasticNet = parameters['ElasticNet']
        SVR = parameters['SVR']
        SVR_RFE = parameters['SVR_RFE']
        RF = parameters['RF']
        LightGBM = parameters['LightGBM']
        XGBoost = parameters['XGBoost']
        MIC = parameters['MIC']
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
        
    try:
        picture_names = SHAP(inputCSV, feature_start, feature_end, target, single, interaction)
    except Exception as e:
        _add_error_xml("Shap Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        raw_data = pd.read_csv(inputCSV)
        y = raw_data[target]
        X = raw_data.iloc[:, feature_start-1: feature_end]
        feature_name = list(X.columns)
        feature_selection_result = feature_selection_regression(X,y,feature_name,10, Pearson, Variance, MIC, Lasso, ElasticNet, SVR,\
                                    SVR_RFE, RF, LightGBM, XGBoost)
    except Exception as e:
        _add_error_xml("Features Vote Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    try:
        _add_info_xml(picture_names, feature_selection_result)
    except Exception as e:
        _add_error_xml("XML Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return

    with open('log.txt', 'a') as fp:
        fp.write('finish\n')
    
if __name__ == '__main__':
    main()