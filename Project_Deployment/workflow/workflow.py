import json
import os
import sys
from time import sleep

import requests
print("into workflow")
path_module_main = os.getcwd()
print(path_module_main)
path_workflow = 'D:\\Workflow\\'
path_module_a = 'D:\\micro\\'
path_module_b = 'D:\\rescu\\'
execute_a = 'md_sample.py'
execute_b = 'workflow_rescu.py'
change_xml_input = 'resultXml.py'
command_change_xml_input = 'python resultXml.py'


addfile_a = 'POSCAR'
addfile_b = 'job'

file_b_a = ''
file_b_b = ''

file_a = ''

para_a = open('parameters1.txt','w')
para_b = open('parameters2.txt','w')
with open('parameters.txt','r') as fin:
    for line in fin.readlines():
        if 'cluster' in line:
            para_a.write(line)
            para_b.write(line)
        if 'input' in line:
            para_a.write(line)
            # file_a = 'microMD_input.tar.gz'
            file_a = line.split(':')[1][:-1]
#para_b.write(f'input:input.tar.gz\n')
para_b.write(f'input:job.zip\n')
para_a.close()
para_b.close()

with open('log.txt', 'a') as fp:
    fp.write('running\n')

taskids = []
with open('taskids.txt','r',encoding = 'utf-8') as fp:
    for line in fp.readlines():
        taskids.append(line.split(':')[1][:-1])

print(file_a)
command_a = 'python md_sample.py ' + file_a
print(command_a)
command_b = 'python workflow_rescu.py '+ taskids[1]

# task 1
os.system(f'md {path_module_a}workdir\\{taskids[0]} && copy parameters1.txt {path_module_a}workdir\\{taskids[0]}\\parameters.txt')

os.system(f'mkdir {path_module_b}workdir\\{taskids[1]}')
#把D:\\rescu\\workdir\\parameters2.txt拷贝到D:\\rescu\\workdir\\taskids[1]目录下
os.system(f'copy parameters2.txt {path_module_b}workdir\\{taskids[1]}\\parameters.txt')

os.system(f'copy {path_module_a}\\workdir\\{execute_a} {path_module_a}workdir\\{taskids[0]} && copy {path_module_a}workdir\\result.xml {path_module_a}workdir\\{taskids[0]}')
os.system(f'copy {file_a} {path_module_a}workdir\\{taskids[0]} && copy {path_module_a}workdir\\{change_xml_input} {path_module_a}workdir\\{taskids[0]}')
print("start run micro.py")
print(taskids[0])
print(command_a)
os.system(f'cd {path_module_a}workdir\\{taskids[0]} && {command_change_xml_input} && {command_a}')


#rint("into micro dictionary")
#last_line = 'running'
#log.txt中的最后一行是finish的时候执行后续操作
#while(last_line != 'finish'):
    #with open(f'{path_module_main}workdir\\{task_id}\\log.txt','r',encoding = 'utf-8') as fp:
#    with open(f'{path_module_a}workdir\\{taskids[0]}\\log.txt','r',encoding = 'utf-8') as fp:
#        lines = fp.readlines()
#        last_line = lines[-1]
#        fp.close()
#解压D:\\micro\\workdir\\taskids[0]\\result.tar.gz文件到D:\\micro\\workdir\\taskids[0]目录下
#os.system(f'7z x {path_module_a}workdir\\{taskids[0]}\\result.tar.gz')
#os.system(f'7z x {path_module_a}workdir\\{taskids[0]}\\result.tar')

#在D:\\rescu\\workdir下新建目录taskid[1]

os.system(f'mkdir {path_module_b}workdir\\{taskids[1]}\\job')
#拷贝D:\\rescu\\workdir\\job和rescu.sh到D:\\rescu\\workdir\\taskids[1]\\result
os.system(f'xcopy {path_module_b}workdir\\{addfile_b} {path_module_b}workdir\\{taskids[1]}\\job /E')

os.system(f'copy {path_module_a}workdir\\{taskids[0]}\\{addfile_a} {path_module_b}workdir\\{taskids[1]}\\job')
#压缩{path_module_a}workdir\\{taskids[0]}\\result到{path_module_b}workdir\\{taskids[0]}\\input.zip
os.system(f'7z a {path_module_b}workdir\\{taskids[1]}\\job.zip {path_module_b}workdir\\{taskids[1]}\\job')
#os.system(f'rd/s/q {path_module_a}workdir\\{taskids[0]}\\result')

#把D:\\rescu\\workdir\\parameters2.txt拷贝到D:\\rescu\\workdir\\taskids[1]目录下
os.system(f'copy {path_module_main}workdir\\parameters2.txt {path_module_b}workdir\\{taskids[1]}\\parameters.txt')
#把D:\\rescu\\workdir\\rescu.py, result.xml拷贝到D:\\rescu\\workdir\\taskids[0]目录下
os.system(f'copy {path_module_b}workdir\\{execute_b} {path_module_b}workdir\\{taskids[1]} && copy {path_module_b}workdir\\result.xml {path_module_b}workdir\\{taskids[1]}')
os.system(f'copy {path_module_b}workdir\\log.txt {path_module_b}workdir\\{taskids[1]}')
#在D:\\rescu\\workdir\\taskids[0]目录下执行python rescu.py
print('---------------------------------------')
os.system(f'cd {path_module_b}workdir\\{taskids[1]} && {command_b}')
print(command_b)


last_line = 'running'
#log.txt中的最后一行是finish的时候执行后续操作
#while(last_line != 'finish\n'):
while(last_line != 'finish\n'):
    print('-------------------------1')
    with open(f'{path_module_b}workdir\\{taskids[1]}\\log.txt','r',encoding = 'utf-8') as fp:
        lines = fp.readlines()
        print('-----------------------2')
        last_line = lines[-1]
        print('--------------------------3')
        print(last_line == 'finish\n')
        fp.close()
print('end')
#os.system(f'copy {path_module_b}workdir\\{taskids[1]}\\result.xml .')
os.system(f'copy {path_module_b}workdir\\{taskids[1]}\\result.tar.gz .')
print('everything is end')
with open(f'{path_module_main}\\log.txt', 'a') as fp:
    fp.write('finish\n')

# UID = 'shu'
# TID = sys.argv[1]
# PORT = 7190
#             input_file_name = line.split(':')[1][:-1]
#         if 'cluster' in line:
#             cluster = line.split(':')[1][:-1]
# print(cluster)

# if cluster == 'zq4000':
#     SERVER = 'sc.shu.edu.cn'
# if cluster == 'sunway':
#     SERVER = '41.0.0.2'

# os.system(f'md temp && cd temp && copy ..\\{input_file_name} .\\ && 7z x {input_file_name} && del {input_file_name}')
# if cluster == 'zq4000':
#     os.system('copy ..\\rescu_zq4000.sh .\\temp\\')
# if cluster == 'sunway':
#     os.system('copy ..\\rescu_openmpi.sh .\\temp\\ && copy ..\\rescu_sunway.sh .\\temp\\')
# os.system('cd temp && 7z a job.tar * && 7z a job.tar.gz job.tar && cd .. && copy temp\\job.tar.gz .\\ && rd /s /q temp')

# # upload
# res = requests.post(f'http://{SERVER}:{PORT}', data={'uid': UID, 'tid': TID},
#                     files={'file': open('job.tar.gz', 'rb')})
# msg = json.loads(res.content.decode('utf-8'))['msg']
# if 'Upload Success' not in msg:
#     with open('log.txt', 'a') as fp:
#         fp.write('killed\n')
#     print('upload error')
#     exit(1)

# # delete temp job.tar.gz
# os.system('del job.tar.gz')

# # submit
# if cluster == 'zq4000':
#     res = requests.get(f'http://{SERVER}:{PORT}/submit?uid={UID}&tid={TID}&exec=rescu_zq4000.sh')
# if cluster == 'sunway':
#     res = requests.get(f'http://{SERVER}:{PORT}/submit?uid={UID}&tid={TID}&exec=rescu_sunway.sh')
# msg = json.loads(res.content.decode('utf-8'))['msg']
# jid = msg
# with open('log.txt', 'a') as fp:
#     fp.write('running\n')
# print(jid)

# # query
# while True:
#     try:
#         res = requests.get(f'http://{SERVER}:{PORT}/query?jid={jid}')
#         msg = json.loads(res.content.decode('utf-8'))['msg']
#         print(msg)
#         # task finished
#         if msg in ['DONE', 'NOMSG']:
#             # get result and tar
#             os.system(f'wget "http://{SERVER}:{PORT}/download?uid={UID}&tid={TID}" -O result.tar.gz')
#             sleep(10)
#             with open('log.txt', 'a') as fp:
#                 fp.write('finish\n')
#             break
#         # task killed
#         if msg == 'EXIT':
#             with open('log.txt', 'a') as fp:
#                 fp.write('killed\n')
#             break
#     except Exception as e:
#         pass
#     sleep(60)
