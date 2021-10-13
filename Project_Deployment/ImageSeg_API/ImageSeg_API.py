import os

Input_filename = ''
Neighborhood_value = ''

#Read xml file
f = open('parameters.txt','r')
for line in f:
    # Parameter
    if "Picture" in line:
        Input_filename = line.split(':')[1][:-1]
    if "Neighborhood_value" in line:
        Neighborhood_value = line.split(':')[1][:-1]
f.close()

path_now = os.getcwd()
command_cp_classify = 'copy D:\\taskschedule\\workdir\\classify.py '+path_now
command_cp_model = 'copy D:\\taskschedule\\workdir\\model(300)(nonNor).txt '+path_now

os.system(command_cp_classify)
os.system(command_cp_model)

#Path of Application
app = '..\\ImageFeatureExtractConsole.exe'

#Run segment application
command = app + ' ' + os.path.join(os.getcwd(), Input_filename) + ' ' + Neighborhood_value
print(command)

with open('log.txt', 'a') as fp:
    fp.write('running\n')

os.system(command)

with open('log.txt', 'a') as fp:
    fp.write('finish\n')

os.system('del classify.py')
os.system('del model(300)(nonNor).txt')