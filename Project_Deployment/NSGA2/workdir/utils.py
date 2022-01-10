import os

from numpy import double
module_name = 'NSGA2'
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
    # inputCSV
    parameters['models'] = list(re_input(parameters['models']).split(','))
            
    parameters['parameter'] = split2int(re_input(parameters['parameter']))
    parameters['start_index'] = int(parameters['start_index'])
    parameters['end_index'] = int(parameters['end_index'])
    parameters['target_index'] = split2int(re_input(parameters['target_index']))
    parameters['target_number'] = len(parameters['target_index'])
    parameters['optimization_direction'] = split2int(re_input(parameters['optimization_direction']))
    parameters['dimension'] = parameters['end_index'] - parameters['start_index'] + 1
    parameters['feature_types'] = split2int(re_input(parameters['feature_types']))
    parameters['feature_lower'] = split2int(re_input(parameters['feature_lower']))
    parameters['feature_upper'] = split2int(re_input(parameters['feature_upper']))
    parameters['is_lower'] = split2int(re_input(parameters['is_lower']))
    parameters['is_upper'] = split2int(re_input(parameters['is_upper']))    
    parameters['population_number'] = int(parameters['population_number'])
    parameters['max_ge'] = int(parameters['max_ge'])
    parameters['cross_possibility'] = float(parameters['cross_possibility'])
    parameters['weight'] = float(parameters['weight'])
    parameters['preference'] = split2int(re_input(parameters['preference'])) 
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
    if system_plat == "Linux":
        os.system('cp ./result.xml ./result_before.xml')  # Linux
    else:
        os.system('copy .\\result.xml .\\result_before.xml')    # Win

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
    if picture_names:
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