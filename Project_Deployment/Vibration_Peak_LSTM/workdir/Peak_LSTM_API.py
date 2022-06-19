import os
import sys
import tarfile

# Get task id
# TID = sys.argv[1]
module_name = 'Peak_LSTM'

# copy shap_func.py
path_now = os.getcwd()
# command_cp_1 = f'copy D:\\{module_name}\\workdir\\{module_name}.py '+ path_now  # Windows
# command_cp_2 = f'copy D:\\{module_name}\\workdir\\utils.py '+ path_now  # Windows
command_cp_1 = f'cp ../{module_name}.py '+ path_now  # Linux
command_cp_2 = f'cp ../utils.py '+ path_now  # Linux
os.system(command_cp_1)
os.system(command_cp_2)

command = f'python {module_name}.py'
with open('log.txt', 'a') as fp:
    fp.write('running\n')

os.system(command)

# os.system(f'del {module_name}.py')  # Windows
os.system(f'rm {module_name}.py')  # Linux
# os.system('del utils.py')  # Windows
os.system('rm utils.py')  # Linux

with tarfile.open('./result.tar.gz', 'w:gz') as tar:
    for filename in os.listdir():
        if filename in ['result.xml', 'parameters.json', 'log.txt', f'{module_name}_API.py']:
            continue
        tar.add(filename)