import matplotlib.pyplot as plt
import re
import json
import numpy as np
#src是predict_Y.out所在位置
def getPoint(src):
    pattern_s = re.compile(r'[A-Z][a-z]+ \(')
    pattern_e = re.compile(r'[A-Z][a-z]+ [A-Z]+')

    with open(src,'r',encoding='utf8') as f:
        line=f.readlines()
    left=0
    right=0
    while not pattern_s.search(line[left]):
        left+=1
    right=left
    res={}
    y_p=[]
    dim=1
    while right<len(line):
        if pattern_s.search(line[right]):
            right+=1
        elif pattern_e.search(line[right]):
            left=right+1
            right=left
            res[dim]=y_p
            y_p=[]
            dim+=1
        else:
            tmp=[]
            for each in line[right].split()[:2]:
                tmp.append(float(each))
            y_p.append(tmp)
            right+=1
    return res,dim

#picdir是图片保存的目录，图片的路径为picdir/nD picture.jpg
def getPic(res,dim,picdir):
    for i in range(1,dim):
        x=np.array(res[i])
        _x=x[:,0]
        y=_x
        _y=x[:,1]
        plt.figure()
        plt.plot(_x,y,linewidth=0.5)
        plt.scatter(_x,_y,s=5,c='r')
        plt.savefig(picdir+'\\'+str(i)+'D picture.jpg')
#示例
src='C:\\Users\\86180\\Desktop\\predict_Y.out'
picdir='C:\\Users\\86180\\Desktop\\pic'
res,dim=getPoint(src)
getPic(res,dim,picdir)