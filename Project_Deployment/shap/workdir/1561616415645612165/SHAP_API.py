import os
import sys

# Get task id
TID = sys.argv[1]

# copy shap_func.py
path_now = os.getcwd()
command_cp_shap = 'copy D:\\shap\\workdir\\shap_func.py '+ path_now  # Windows
# command_cp_shap = 'cp ../shap_func.py '+ path_now  # Linux
print(command_cp_shap)
os.system(command_cp_shap)

command = 'python shap_func.py'
with open('log.txt', 'a') as fp:
    fp.write('running\n')

os.system(command)

os.system('del shap_func.py')  # Windows
# os.system('rm shap_func.py')  # Linux

with open('log.txt', 'a') as fp:
    fp.write('finish\n')