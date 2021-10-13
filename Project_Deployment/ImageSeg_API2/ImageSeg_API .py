import os

Input_filename = ''

#Read xml file
f = open('parameters.txt','r')
parameters = []
for line in f:
    # Parameter
    if "Picture" in line:
        Input_filename = line.split(':')[1][:-1]
    if "Neighborhood_value" in line:
        Neighborhood_value = line.split(':')[1][:-1]
    parameters.append(line.strip().split(':')[1])
f.close()

pstring = (' ').join(parameters[1:-1]) + ' 1 ' + parameters[-1]

path_now = os.getcwd()
command_cp_kmeans = 'copy D:\\taskschedule2\\workdir\\kmeans.py '+path_now
os.system('md results')
os.system(command_cp_kmeans)

#Path of Application
app = '..\\MS.exe'

#Run segment application
command = app + ' ' + os.path.join(os.getcwd(), Input_filename) + ' ' + pstring
print(command)

with open('log.txt', 'a') as fp:
    fp.write('running\n')

os.system(command)
with open('results/result.txt','r') as fin:
    nums = []
    line = fin.read().split()
    for i in line:
        if len(i)<=2:
            nums.append(i)
    string_xml = 'There are ' + nums[0] + ' types of metallographic in this image: ' + nums[4] + ' in the first category; '+nums[8]+' in the second category;'
    fxml_in = open('result.xml','r')
    result_xml = fxml_in.read()
    result_xml = result_xml.replace('<Result_text>{Result_text}</Result_text>', f'<Result_text>{string_xml}</Result_text>')
    fxml_out = open('result.xml','w')
    fxml_out.write(result_xml)    
os.system('copy results\\result_clustering.png .')
os.system('rd/s/q results')
with open('log.txt', 'a') as fp:
    fp.write('finish\n')

os.system('del kmeans.py')