import os
import sys

# Get task id
TID = sys.argv[1]

# copy shap_func.py
path_now = os.getcwd()
command_cp_shap = 'copy D:\\feature_selection\\workdir\\feature_selection.py '+ path_now  # Windows
# command_cp_shap = 'cp ../feature_selection.py '+ path_now  # Linux
os.system(command_cp_shap)

command = 'python feature_selection.py'
with open('log.txt', 'a') as fp:
    fp.write('running\n')

os.system(command)

with open('log.txt', 'a') as fp:
    fp.write('finish\n')

os.system('del feature_selection.py')  # Windows
# os.system('rm feature_selection.py')  # Linux