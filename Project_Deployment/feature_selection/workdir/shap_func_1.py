import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
import shap
# %matplotlib inline

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
    try:
        parameters['single'] = split2int(re_input(parameters['single']))
    except:
        parameters['single'] = None
    try:
        parameters['interactionA'] = split2int(re_input(parameters['interactionA']))
        parameters['interactionB'] = split2int(re_input(parameters['interactionB']))
        parameters['interaction'] = [[A,B] for A,B in zip(parameters['interactionA'], parameters['interactionB'])]
    except:
        parameters['interaction'] = None
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

    shap_values = explainer.shap_values(data[features])

    shap.initjs()

    picture_names = {}

    summary_names = []
    shap.summary_plot(shap_values, data[features], show=False)
    name = 'summary_shap_values.png'
    summary_names.append(name)
    plt.savefig('./'+name)
    plt.close()

    shap.summary_plot(shap_values, data[features], plot_type="bar", show=False)
    name = 'summary_importance_values.png'
    summary_names.append(name)
    plt.savefig('./'+name)
    plt.close()

    picture_names['summary_values'] = summary_names


    if single:
        single_names = []
        for i in single:
            shap.dependence_plot(features[i], shap_values, data[features], interaction_index=None, show=False)
            name = 'shap_values_'+features[i]+'.png'
            single_names.append(name)
            plt.savefig('./'+name)
            plt.close()
        picture_names['single_feature'] = single_names


    #shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[features])
    if interaction:
        interaction_names = []
        for interaction_A, interaction_B in interaction:
            shap.dependence_plot(features[interaction_A], shap_values, data[features], interaction_index=features[interaction_B], show=False)
            name = 'interaction_values_'+features[interaction_A]+'_inter_'+features[interaction_B]+'.png'
            interaction_names.append(name)
            plt.savefig('./'+name)
            plt.close()
        picture_names['interaction_feature'] = interaction_names

    return picture_names


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


def _add_pic_xml(picture_names) -> None:
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

    element_picture = ET.Element('picture')
    for picture_name in picture_names:
        element_file, element_name, element_url = ET.Element('file'), ET.Element('name'), ET.Element('url')
        element_name.text, element_url.text = picture_name, picture_name
        element_picture.append(element_file)
        element_file.append(element_name)
        element_file.append(element_url)
    output.append(element_picture)
    __indent(root)
    ET.tostring(root, method='xml')
    # Save 
    tree.write('result.xml', encoding='utf-8', xml_declaration=True)
    with open('result.xml', 'r') as fp:
        lines = [line for line in fp]
        lines.insert(1, '<?xml-stylesheet type="text/xsl" href="/XSLTransform/SHAP.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))


def main():
    parameters = _read_parameters()
    inputCSV = parameters['inputCSV']
    feature_start = parameters['feature_start']
    feature_end = parameters['feature_end']
    target = parameters['target']
    single = parameters['single']
    interaction = parameters['interaction']
    picture_names = SHAP(inputCSV, feature_start, feature_end, target, single, interaction)
    _add_pic_xml(picture_names)
    

if __name__ == '__main__':
    main()