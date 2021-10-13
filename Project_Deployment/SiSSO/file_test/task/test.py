import os

input_file_name = 'test.dat'

os.system(f'md temp && cd temp && copy ..\\{input_file_name} .\\train.dat')
# copy .sh file to workdir
os.system('copy ..\\SISSO.sh .\\temp\\')
# copy data file to workdir
os.system('copy ..\\SISSO.in_model .\\ &&  copy ..\\replace.py .\\')
# make SISSO.in file
os.system('python replace.py && copy .\\SISSO.in .\\temp\\')
# tar job.tar.gz
os.system('cd temp && 7z a job.tar * && 7z a job.tar.gz job.tar && cd .. && copy temp\\job.tar.gz .\\ && rd /s /q temp')