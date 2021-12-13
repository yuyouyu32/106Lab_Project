module_name = 'arima'
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
    # 2 CSV Read...
    parameters['features'] = list(range(int(parameters['feature_start_index']) - 1,int(parameters['feature_end_index']), 1))
    parameters['target'] = split2int(re_input(parameters['target_index']))
    for index, item in parameters['target']:
        parameters['target'][index] = item - 1
    parameters['seasonal']= True if parameters['seasonal'] == 'True' else False
    # information_criterion Read (aic, bic)
    if parameters['information_criterion'] not in ('bic', 'aic'):
        parameters['information_criterion'] = 'bic'
    try:
        parameters['m'] = int(parameters['m'])
    except:
        parameters['m'] = 5
    try:
        parameters['start_p'] = int(parameters['start_p'])
    except:
        parameters['start_p'] = 2
    try:
        parameters['d'] = int(parameters['d'])
    except:
        parameters['d'] = None
    try:
        parameters['start_q'] = int(parameters['start_q'])
    except:
        parameters['start_q'] = 2
    try:
        parameters['max_p'] = int(parameters['max_p'])
    except:
        parameters['max_p'] = 5
    try:
        parameters['max_d'] = int(parameters['max_d'])
    except:
        parameters['max_d'] = 2
    try:
        parameters['max_q'] = int(parameters['max_q'])
    except:
        parameters['max_q'] = 5       
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
    os.system('cp ./result.xml ./result_before.xml')  # Linux
    # os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("./result.xml")
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
    os.system('cp ./result.xml ./result_before.xml')  # Linux
    # os.system('copy .\\result.xml .\\result_before.xml')    # Win

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