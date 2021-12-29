import os
module_name = 'RNN'
def _read_parameters():
    # result = arima('test.csv', feature=[1,2], target=3,file_path="test2.csv", seasonal=False, m=5, information_criterion='bic', start_p=2, d=None,
    #       start_q=2, max_p=5, max_d=2, max_q=5)
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
        parameters['look_back'] = int(parameters['look_back'])
    except:
        parameters['look_back'] = 4
    try:
        parameters['predict_num'] = int(parameters['predict_num'])
    except:
        parameters['predict_num'] = 10
    try:
        parameters['hidden_dim'] = int(parameters['hidden_dim'])
    except:
        parameters['hidden_dim'] = 64
    try:
        parameters['layer_dim'] = int(parameters['layer_dim'])
    except:
        parameters['layer_dim'] = 3
    try:
        parameters['batch_size'] = int(parameters['batch_size'])
    except:
        parameters['batch_size'] = 4
    try:
        parameters['dropout'] = float(parameters['dropout'])
    except:
        parameters['dropout'] = 0.2
    try:
        parameters['n_epochs'] = int(parameters['n_epochs'])
    except:
        parameters['n_epochs'] = 100
    try:
        parameters['learning_rate'] = float(parameters['learning_rate'])
    except:
        parameters['learning_rate'] = 1e-3
    try:
        parameters['weight_decay'] = float(parameters['weight_decay'])
    except:
        parameters['weight_decay'] = 1e-6
    
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


def _add_info_xml(picture_names, result) -> None:
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import os
    # Back up result.xml
    # os.system('cp ./result.xml ./result_before.xml')  # Linux
    os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("../result.xml")
    root = tree.getroot()
    output = tree.find('output')

    # Table
    element_result = ET.Element('Result')
    element_table = ET.Element('table')
    for index, row_data in result.iterrows():
        element_row = ET.Element('row')
        element_index = ET.Element('index')
        element_index.text = str(index + 1)
        element_row.append(element_index)
        for column_name, value in zip(row_data.keys(), row_data.values):
            element_column = ET.Element(column_name)
            try:
                element_column.text = str(round(value, 3))
            except:
                element_column.text = str(value)
            element_row.append(element_column)
        element_table.append(element_row)
    element_result.append(element_table)
    output.append(element_result)

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