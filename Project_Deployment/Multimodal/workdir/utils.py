import os

from numpy import double
module_name = 'Multimodal'
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
    
    parameters['start_index'] = int(parameters['start_index'])
    parameters['end_index'] = int(parameters['end_index'])
    parameters['target_name'] = parameters['target_name']
    parameters['learning_rate'] = float(parameters['learning_rate'])
    parameters['epochs'] = int(parameters['epochs'])
        
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


def _add_info_xml(picture_names, score) -> None:
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

    
    # Score
    element_score = ET.Element('Score')
    element_best_score = ET.Element('R2')
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