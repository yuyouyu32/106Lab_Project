import json
import os
import sys
from time import sleep
import requests

UID = 'shu'
TID = sys.argv[1]
PORT = 7190

# create job.tar.gz
with open('parameters.txt', 'r') as fp:
    for line in fp.readlines():
        if 'train_data' in line:
            input_file_name = line.split(':')[1][:-1]
        if 'cluster' in line:
            cluster = line.split(':')[1][:-1]
print(cluster)

SERVER = 'sc.shu.edu.cn'


os.system(f'md temp && cd temp && copy ..\\{input_file_name} .\\train.dat')
# copy .sh file to workdir
os.system('copy ..\\SISSO.sh .\\temp\\')
# copy data file to workdir
os.system('copy ..\\SISSO.in_model .\\ &&  copy ..\\replace.py .\\  &&  copy ..\\DataProcessing.py .\\')
# make SISSO.in file
os.system('python replace.py && copy .\\SISSO.in .\\temp\\')
# tar job.tar.gz
os.system('cd temp && 7z a job.tar * && 7z a job.tar.gz job.tar && cd .. && copy temp\\job.tar.gz .\\ && rd /s /q temp')

# upload
res = requests.post(f'http://{SERVER}:{PORT}', data={'uid': UID, 'tid': TID},
                    files={'file': open('job.tar.gz', 'rb')})
msg = json.loads(res.content.decode('utf-8'))['msg']
if 'Upload Success' not in msg:
    with open('log.txt', 'a') as fp:
        fp.write('killed\n')
    print('upload error')
    exit(1)

# delete temp job.tar.gz
os.system('del job.tar.gz')

# submit
res = requests.get(f'http://{SERVER}:{PORT}/submit?uid={UID}&tid={TID}&exec=SISSO.sh')
msg = json.loads(res.content.decode('utf-8'))['msg']
jid = msg
with open('log.txt', 'a') as fp:
    fp.write('running\n')
print(jid)

# query
while True:
    try:
        res = requests.get(f'http://{SERVER}:{PORT}/query?jid={jid}')
        msg = json.loads(res.content.decode('utf-8'))['msg']
        print(msg)
        # task finished
        if msg in ['DONE', 'NOMSG']:
            # get result and tar
            os.system(f'wget "http://{SERVER}:{PORT}/download?uid={UID}&tid={TID}" -O result.tar.gz')
            os.system(f'7z x result.tar.gz && 7z x result.tar && move .\\{TID}\\desc_dat .\\ && move .\\{TID}\\SISSO.out .\\')
            os.system('python DataProcessing.py')
            picfiles = os.listdir('.\\picture')
            for picfile in picfiles:
                os.system(f'cd .\\picture && copy {picfile} ..\\')
            os.system(f'rd /s /q {TID}')
            os.system(f'rd /s /q desc_dat')
            os.system(f'rd /s /q picture')
            sleep(10)
            with open('log.txt', 'a') as fp:
                fp.write('finish\n')
            break
        # task killed
        if msg == 'EXIT':
            with open('log.txt', 'a') as fp:
                fp.write('killed\n')
            break
    except Exception as e:
        pass
    sleep(60)
