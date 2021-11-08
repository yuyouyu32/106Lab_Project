from selenium import webdriver
import json
import time
import random
from selenium.webdriver.common.keys import Keys
import xlrd
import os
import sys


def Url_spider(inputname,UserID='2913448338@qq.com',Password='1234567lj'):
    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory": os.getcwd()}
    chromeOptions.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(chrome_options=chromeOptions)
    driver.get("http://www.matweb.com/membership/login.aspx")
    login_id = driver.find_element_by_xpath("//*[@id='ctl00_ContentMain_txtUserID']")
    login_id.send_keys(UserID)
    login_password = driver.find_element_by_xpath('//*[@id="ctl00_ContentMain_txtPassword"]')
    login_password.send_keys(Password)
    login_password.send_keys(Keys.ENTER)
    driver.get("http://www.matweb.com/search/MaterialGroupSearch.aspx")
    driver.find_element_by_xpath("//*[@id='ctl00_ContentMain_ucMatGroupTree_msTreeViewn1']/img").click()
    # driver.find_element_by_xpath("//*[@id='ctl00_ContentMain_ucMatGroupTree_msTreeViewn19']/img").click()
    time.sleep(3)
    j = 0
    print('start')
    for i in range(8, 23):
        element_name = driver.find_element_by_id(f"ctl00_ContentMain_ucMatGroupTree_msTreeViewt{i}").get_attribute(
            "textContent")
        print(element_name)
        if element_name.find(inputname) + 1:
            j = i
            print(j)
            break
    driver.find_element_by_id(f"ctl00_ContentMain_ucMatGroupTree_msTreeViewt{j}").click()
    driver.find_element_by_xpath("//*[@id='ctl00_ContentMain_btnSubmit']").click()
    driver.implicitly_wait(3)
    href_list = {}
    total = driver.find_element_by_xpath("//*[@id='ctl00_ContentMain_UcSearchResults1_lblPageTotal']").get_attribute(
        "textContent")
    total = int(total)
    if total > 30:
        total = 30
    temp = []
    i = 1
    while i <= total:
        time.sleep(random.randint(3, 5))
        for link in driver.find_elements_by_xpath("//*[@id='tblResults']/tbody//a"):
            if link.get_attribute('textContent') == "Material Name":
                continue
            else:
                temp.append(link.get_attribute('href'))
        driver.find_element_by_id("ctl00_ContentMain_UcSearchResults1_lnkNextPage").click()
        i = i + 1
    for base_url in temp:
        download_url = base_url.replace('DataSheet', 'CreateIQY')
        driver.get(download_url)
        time.sleep(1)
    driver.close()

def excel2json(filename):
    file=xlrd.open_workbook(filename)
    table=file.sheets()[0] #读取表单
    nrows=table.nrows  #有效行数
    #单元格数据类型：0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
    merged=table.merged_cells
    i=0
    data={}
    while i < nrows:
        if table.row(i)[1].value == "Metric":
            dict_name = table.row(i)[0].value.split(" ")[0]  # 取空格前字符
            globals()[dict_name] = {}  # 动态创建字典变量
            i += 1                     #取到下一行
            while table.cell_type(i,1) != 0:    #数据单元格类型不为空时
                n = IsMerged(i,0,merged)    #判断是否为合并单元格，返回合并行数，否则为0
                if n:                                                                               #合并单元格
                    sondict_name = table.cell_value(i,0).replace(" ","_").replace(",","_")             #替换逗号和空格
                    globals()[sondict_name] = {}
                    if n == 2:                                                                      #一行条件时
                        eval(sondict_name)[table.cell_value(i + 1, 1)] = table.cell_value(i, 1)    #先取第一个数据
                        i += 2
                        try:
                            iscondition = str(table.cell_value(i+1,1))
                            while iscondition.find("@")+1 and table.cell_type(i,0) == 0:    #循环取数据
                                eval(sondict_name)[table.cell_value(i + 1, 1)] = table.cell_value(i, 1)
                                i += 2
                            eval(dict_name)[sondict_name] = eval(sondict_name)
                        except:
                            eval(dict_name)[sondict_name] = eval(sondict_name)
                            break
                    else:                                                                           #多行条件时
                        condition = String(i,n,table)  #条件拼接
                        eval(sondict_name)[condition] = table.cell_value(i,1)
                        i += n
                        try:
                            while str(table.cell_value(i + 1, 1)).find("@")+1 and table.cell_type(i,0) == 0:
                                condition = String(i,n,table)
                                eval(sondict_name)[condition] = table.cell_value(i,1)
                                i += n
                        except:
                            eval(dict_name)[sondict_name] = eval(sondict_name)
                            break
                        eval(dict_name)[sondict_name] = eval(sondict_name)
                else:                                                                            #普通单元格
                    try:
                        if table.cell_type(i+1,0) == 0 and table.cell_type(i+1,1) != 0:                 #if为列表
                            list_name=table.cell_value(i,0).replace(" ","_").replace(",","_")
                            globals()[list_name] = []
                            eval(list_name).append(str(table.cell_value(i,1))+str(table.cell_value(i,3)))
                            i += 1
                            while table.cell_type(i,1) != 0 and table.cell_type(i,0) == 0:
                                eval(list_name).append(str(table.cell_value(i,1))+str(table.cell_value(i,3)))
                                i += 1
                            eval(dict_name)[list_name]=eval(list_name)
                        else:
                            eval(dict_name)[table.cell_value(i, 0)] = table.cell_value(i, 1)
                            i += 1
                    except:
                        data[dict_name] = eval(dict_name)
                        break
            data[dict_name]=eval(dict_name)
        else:
            i += 1
    print(data)
    jsonname = f'{filename}.json'.replace(".xlsx","")
    with open(jsonname,'w') as f:
        json.dump(data,f)
    data={}

def IsMerged(row,col,merged):
    for (rlow, rhigh, clow, chigh) in merged:  # 遍历表格中所有合并单元格位置信息
        if (row >= rlow and row < rhigh):  # 行坐标判断
            if (col >= clow and col < chigh):  # 列坐标判断
                n=rhigh-rlow
                return n
            else:
                continue
        else:
            continue
    return 0

def String(i,n,table):
    temp = "".join(str([table.cell_value(i+j,1) for j in range(1, n)]))
    return temp


def _read_parameters():
    with open('parameters.json','r',encoding='utf8')as fp:
        parameters = json.load(fp)
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
        inputXLSX = parameters['inputXLSX']
        element_name = parameters['element_name']
    except Exception as e:
        _add_error_xml("Parameters Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    try:
        if element_name != 'None':
            Url_spider(element_name)
    except Exception as e:
        _add_error_xml("Spider Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    try:
        excel2json(inputXLSX)
    except Exception as e:
        _add_error_xml("Excel to Json Error", str(e))
        with open('log.txt', 'a') as fp:
            fp.write('error\n')
        return
    

    # Url_spider('Aerogel')

if __name__ == '__main__':
    console = sys.stdout              
    file = open("task_log.txt", 'w')
    sys.stdout = file                   				 	
    main()
    sys.stdout = console
    file.close()

