import os
import platform
import subprocess
import sys
import tarfile


# Get task id
TID = sys.argv[1]
module_name = 'Original_SISSO'
system_plat = platform.system()

# copy shap_func.py
path_now = os.getcwd()
if system_plat == "Linux":
    command_cp_1 = f'cp ../SISSO.in '+ path_now  # Linux
    command_cp_2 = f'cp ../utils.py '+ path_now  # Linux
else:
    command_cp_1 = f'copy ..\\SISSO.in '+ path_now  # Windows
    command_cp_2 = f'copy ..\\utils.py '+ path_now  # Windows
    
os.system(command_cp_1)
os.system(command_cp_2)

from utils import _add_error_xml, _add_info_xml, _read_parameters, _get_result, draw_pics
Flag = True
# Read Parameters
try:
    _read_parameters()
except Exception as e:
    _add_error_xml("Parameters Error", str(e))
    Flag = False
    with open('log.txt', 'a') as fp:
        fp.write('error\n')
    
# SISSO excute
command = '/home/lab106/WorkSpace/SISSO/SISSO2'
command_predict = '/home/lab106/WorkSpace/SISSO/SISSO_predict'
finish = False
if Flag:
    with open('log.txt', 'a') as fp:
        fp.write('running\n')
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    ret_pre = subprocess.run(command_predict,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    if ret.returncode == 0 and ret_pre.returncode == 0:
        try:
            _add_info_xml(_get_result('SISSO.out'), draw_pics())
            finish = True
        except Exception as e:
            _add_error_xml("XML Error", str(e))
            with open('log.txt', 'a') as fp:
                fp.write('error\n') 
    else:
        _add_error_xml("SISSO run time error", 'Please check your input .dat file and other parameters.')
    
# Delete file
if system_plat == "Linux":
    # os.system(f'rm {module_name}.py')  # Linux
    os.system('rm utils.py')  # Linux
else:
    # os.system(f'del {module_name}.py')  # Window
    os.system('del utils.py')  # Windows

# add tar file
with tarfile.open('./result.tar.gz', 'w:gz') as tar:
    for filename in os.listdir():
        if filename in ['result.xml', 'parameters.json', 'log.txt', f'{module_name}_API.py']:
            continue
        tar.add(filename)
        
if finish:
    with open('log.txt', 'a') as fp:
        fp.write('finish\n')
elif Flag and not finish:
    with open('log.txt', 'a') as fp:
        fp.write('error\n')
