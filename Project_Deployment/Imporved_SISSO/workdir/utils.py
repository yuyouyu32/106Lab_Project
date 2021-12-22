module_name = 'ISISSO'
import platform
system_plat = platform.system()
import json
import os
from collections import deque
import re



def _get_result(pathsrc):
    dic_f={}
    dic_c={}
    dic_b={}
    dic_rmse={}
    dic_mae={}
    pattern_f = re.compile(r' +[0-9]+:\[')
    pattern_c = re.compile(r' +[a-z]+_[0-9]+:')
    pattern_b = re.compile(r' +[A-Z]+[a-z]+_[0-9]+:')
    pattern_r = re.compile(r' +[A-Z]+,[A-Z]+[a-z]+[A-Z]+_[0-9]+:')
    pattern_d = re.compile(r' +[0-9][A-Z]+ +[a-z]+ +[(]?[a-z]+[)]?:')
    feature=deque()
    coeffic=deque()
    bias=deque()
    rmse=deque()
    dim=[]
    with open(pathsrc,'r',encoding='utf8')as f:
        for line in f:
            g = pattern_f.search(line)
            h = pattern_c.search(line)
            b = pattern_b.search(line)
            r = pattern_r.search(line)
            d = pattern_d.search(line)
            if g:
                feature.append(line.split(':')[1].strip())
            if h:
                coeffic.append(line.split()[1:])
            if b:
                bias.append(line.split()[1])
            if r:
                rmse.append(line.split()[1:])
            if d:
                dim.append(int(line.split()[0][0]))      
    for i in range(max(dim)):
        tmp=[]
        for j in range(i+1):
            tmp.append(feature.popleft())
        dic_f["Formula "+str(i+1)+"D"]=tmp
        dic_c["Formula "+str(i+1)+"D"]=coeffic.popleft()
        dic_b["Formula "+str(i+1)+"D"]=bias.popleft()
        tmpr=rmse.popleft()
        dic_rmse[str(i+1)+"D"]=tmpr[0]
        dic_mae[str(i+1)+"D"]=tmpr[1]
    res={}
    res['features']=dic_f
    res['coefficients']=dic_c
    res['bias']=dic_b
    res['RMSE']=dic_rmse
    res['MaxAE']=dic_mae
    return res
        
        
def _read_parameters():
    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
    nsf = int(parameters['nsf'])
    rung = int(parameters['rung'])
    nsample = int(parameters['nsample'])
    desc_dim = int(parameters['desc_dim'])
    maxcomplexity = int(parameters['maxcomplexity'])
    dimclass = parameters['dimclass']
    subs_sis = int(float(parameters['subs_sis']))
    with open('SISSO.in', 'r') as f:
        sisso_in = f.read()
    with open('SISSO.in', 'w') as f:
        f.write(sisso_in.format(nsf=nsf, rung=rung, nsample=nsample, desc_dim=desc_dim, maxcomplexity=maxcomplexity, dimclass=dimclass, subs_sis=subs_sis))
    train_data = parameters['train_data']
    if system_plat == "Linux":
        os.system(f'mv ./{train_data} ./train.dat')  # Linux
    else:
        os.system(f'ren .\\{train_data} train.dat')    # Win

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


def _add_info_xml(result) -> None:
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    # Back up result.xml
    if system_plat == "Linux":
        os.system('cp ./result.xml ./result_before.xml')  # Linux
    else:
        os.system('copy .\\result.xml .\\result_before.xml')    # Win

    tree = ET.parse("./result.xml")
    root = tree.getroot()
    output = tree.find('output')
    
    element_result = ET.Element('Result')
    for key, values in result.items():
        if key in {"features", "coefficients"}:
            element_key = ET.Element(key)
            for dim, value_dim in values.items():
                element_dim = ET.Element(dim)
                for value in value_dim:
                    element_value = ET.Element('value')
                    element_value.text = str(value)
                    element_dim.append(element_value)
                element_key.append(element_dim)
        else:
            element_key = ET.Element(key)
            for dim, value_dim in values.items():
                element_dim = ET.Element(dim)
                element_value = ET.Element('value')
                element_value.text = str(value_dim)
                element_dim.append(element_value)
                element_key.append(element_dim)
        element_result.append(element_key)
    output.append(element_result)
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