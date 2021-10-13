#coding=utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#获取维度
dim = 0
with open('SISSO.in','r') as fin:
    alllines = fin.readlines()
    list = []
    for line in alllines:
        if 'desc_dim=' in line:
            dim = int(line[9])

#获取公式
RMSE = []
MaxAE = []
formula  = []
R2 = []
cnt = 0
with open('SISSO.out','r') as fin:
    alllines = fin.readlines()
    list = []
    dimcnt=1
    for i in range(0,len(alllines)):
        if 'descriptor (model)' in alllines[i]:
            parameters = []
            k = []
            p = []
            for j in range(0,dimcnt):
                parameters.append(alllines[i+3+j].strip().split()[0].split(':')[1])
            print(parameters)
            for j in range(i + 0, i + 7 + cnt):
                if 'coefficients_' in alllines[j]:
                    indexk = alllines[j].strip().split()
                    del indexk[0]
                    k = indexk
                if 'Intercept_' in alllines[j]:
                    p.append(alllines[j].strip().split(':')[1].strip())
            formula_index = str(dimcnt)+'D: '+'y = '
            for j in range(0,dimcnt):
                formula_index+='('+k[j]+')*'+parameters[j]+'+'
            formula_index+='('+p[0]+')'
            formula.append(formula_index)
            dimcnt += 1
            cnt+=1
        if 'RMSE,MaxAE_' in alllines[i]:
            data = alllines[i].strip().split()
            RMSE.append(data[1])
            MaxAE.append(data[2])

#绘制图片
os.system('md picture')
path = 'desc_dat'
picfiles = os.listdir(path)
picturename = []
for file in picfiles:
    measure = []
    fitting = []
    with open(path+'/'+file,'r') as fin:
        alllines = fin.readlines()
        del alllines[0]
        for line in alllines:
            measure.append(float(line.strip().split()[1]))
            fitting.append(float(line.strip().split()[2]))
        R2.append('%.15f'%(r2_score(measure,fitting)))
        a = [-1000000,10000000]
        lim_min = min(min(measure),min(fitting))
        lim_max = max(max(measure),max(fitting))
        plt.figure(1)
        plt.scatter(measure, fitting, c='red', s=5, alpha=0.6)
        plt.xlim(lim_min,lim_max)
        plt.ylim(lim_min,lim_max)
        plt.plot(a, a, c='green')
        picname = 'picture/' + file.strip('.dat') + '.png'
        picturename.append(picname)
        plt.savefig(picname)
        plt.cla()


#输出每一维的Score结果
score = []
for i in range(0,dim):
    string = str(i+1)+'D:\tR2='+R2[i]+'\tRMSE='+RMSE[i]+'\tMaxAE='+MaxAE[i]
    score.append(string)

for file in picfiles:
    os.system('copy picture\\{file} .\\')

#修改result.xml文件
os.system('copy .\\result.xml .\\result_before.xml')
with open('result_before.xml','r') as fin:
    with open('result.xml','w') as fout:
        alllines = fin.readlines()
        for i in range(0,20):
            fout.write(alllines[i])
        for i in range(0,dim):
            fout.write('    <formula'+str(i)+'>' + formula[i] + '</formula'+str(i)+'>\n')
            fout.write('    <score'+str(i)+'>'+score[i]+'</score'+str(i)+'>\n')
        fout.write('    <picture>\n')
        for i in range(0,dim):
            fout.write('      <file>\n')
            fout.write('        <name>'+picturename[i].split('/')[1]+'</name>\n')
            fout.write('        <url>' + picturename[i].split('/')[1] + '</url>\n')
            fout.write('      </file>\n')
        fout.write('    </picture>\n')
        for i in range(20,len(alllines)):
            fout.write(alllines[i])
