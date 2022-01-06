import os

from numpy import double
module_name = 'FSRL'
import platform
system_plat = platform.system()



def _read_parameters():
    import json
    def re_input(original: str):
        return original.replace('，', ',').replace(' ', '')
    def split2int(original: str):
        return list(map(int, original.split(',')))

    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    
    try:
        parameters['start_index'] = int(parameters['start_index'])
    except:
        parameters['start_index'] = 1
    try:
        parameters['end_index'] = int(parameters['end_index'])
    except:
        parameters['end_index'] = 2
    try:
        parameters['target_index'] = int(parameters['target_index'])
    except:
        parameters['target_index'] = -1
    try:
        parameters['max_choose'] = int(parameters['max_choose'])
    except:
        parameters['max_choose'] = 1
    try:
        parameters['target_score'] = float(parameters['target_score'])
    except:
        parameters['target_score'] = 0.5
    try:
        parameters['parameter'] = int(parameters['parameter'])
    except:
        parameters['parameter'] = 10
    try:
        parameters['epochs'] = int(parameters['epochs'])
    except:
        parameters['epochs'] = 500
        
    if parameters['kind'] == '0':
        parameters['score_kind'] = 'accuracy_score'
        if parameters['model'] == '0':
            parameters['model'] = 'svc'
        elif parameters['model'] == '1':
            parameters['model'] = 'rf_c'
        else:
            parameters['model'] = 'xgboost'
    else:
        parameters['score_kind'] = 'r2_score'
        if parameters['model'] == '0':
            parameters['model'] = 'svr'
        elif parameters['model'] == '1':
            parameters['model'] = 'rf_r'
        else:
            parameters['model'] = 'xgboost'
        
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


def _add_info_xml(picture_names, names, indexs, score) -> None:
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import os
    # Back up result.xml
    if system_plat == "Linux":
        os.system('cp ./result.xml ./result_before.xml')  # Linux
    else:
        os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("./result.xml")
    root = tree.getroot()
    output = tree.find('output')

    # Table
    element_result = ET.Element('Best_feature_table')
    element_table = ET.Element('table')
    
    element_row_1 = ET.Element('row')
    for index in indexs[0]:
        element_column_index = ET.Element('Index')
        element_column_index.text = str(index+1)
        element_row_1.append(element_column_index)
    element_table.append(element_row_1)
    element_row_2 = ET.Element('row')
    for name in names:
        element_column_name = ET.Element('Column_name')
        element_column_name.text = str(name)
        element_row_2.append(element_column_name)
    element_table.append(element_row_2)
    
    element_result.append(element_table)
    output.append(element_result)
    
    # Score
    element_score = ET.Element('Score')
    element_best_score = ET.Element('best_score')
    element_best_score.text = str(round(score, 4))
    element_score.append(element_best_score)
    output.append(element_score)
    
    # Picture
    element_pictures = ET.Element('Pictures')
    for picture_name in picture_names:
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
        lines.insert(1, f'<?xml-stylesheet type="text/xsl" href="/XSLTransform/{module_name}.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))


def _add_error_xml(error_message, error_detail):
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import os
    # Back up result.xml
    if system_plat == "Linux":
        os.system('cp ./result.xml ./result_before.xml')  # Linux
    else:
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
        lines.insert(1, f'<?xml-stylesheet type="text/xsl" href="/XSLTransform/{module_name}.xsl" ?>\n')
    with open('result.xml', 'w') as fp:
        fp.write(''.join(lines))
        


# def main():
#     try:
#         ...
#     except Exception as e:
#         _add_error_xml("Parameters Error", str(e))
#         with open('log.txt', 'a') as fp:
#             fp.write('error\n')
#         return

#     try:
#        ...
#     except Exception as e:
#         _add_error_xml("xxx Error", str(e))
#         with open('log.txt', 'a') as fp:
#             fp.write('error\n')
#         return

#     try:
#         _add_info_xml(...)
#     except Exception as e:
#         _add_error_xml("XML Error", str(e))
#         with open('log.txt', 'a') as fp:
#             fp.write('error\n')
#         return
  
#     with open('log.txt', 'a') as fp:
#         fp.write('finish\n')

# if __name__ == '__main__':
#     console = sys.stdout
#     file = open("task_log.txt", "w")
#     sys.stdout = file
#     main()
#     sys.stdout = console
#     file.close