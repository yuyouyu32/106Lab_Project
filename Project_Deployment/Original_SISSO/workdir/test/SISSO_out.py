from collections import deque
import re
import json
def tojson(pathsrc,pathtarget):
    dic_f={}
    dic_c={}
    dic_b={}
    dic_rmse={}
    dic_mae={}
    pattern_f = re.compile(r' +[0-9]+:\[')
    pattern_c = re.compile(r' +[a-z]+_[0-9]+:')
    pattern_b = re.compile(r' +[A-Z]+[a-z]+_[0-9]+:')
    pattern_r = re.compile(r' +[A-Z]+,[A-Z]+[a-z]+[A-Z]+_[0-9]+:')
    pattern_d = re.compile(r' +[0-9][A-Z]+ +[a-z]+ +[(]?[a-z]+[)]?:')
    feature=deque()
    coeffic=deque()
    bias=deque()
    rmse=deque()
    dim=[]
    with open(pathsrc,'r',encoding='utf8')as f:
        for line in f:
            g = pattern_f.search(line)
            h = pattern_c.search(line)
            b = pattern_b.search(line)
            r = pattern_r.search(line)
            d = pattern_d.search(line)
            if g:
                feature.append(line.split(':')[1].strip())
            if h:
                coeffic.append(line.split()[1:])
            if b:
                bias.append(line.split()[1])
            if r:
                rmse.append(line.split()[1:])
            if d:
                dim.append(int(line.split()[0][0]))
                

                
    for i in range(max(dim)):
        tmp=[]
        for j in range(i+1):
            tmp.append(feature.popleft())
        dic_f[str(i+1)+"D"]=tmp
        dic_c[str(i+1)+"D"]=coeffic.popleft()
        dic_b[str(i+1)+"D"]=bias.popleft()
        tmpr=rmse.popleft()
        dic_rmse[str(i+1)+"D"]=tmpr[0]
        dic_mae[str(i+1)+"D"]=tmpr[1]
    res={}
    res['features']=dic_f
    res['coefficients']=dic_c
    res['bias']=dic_b
    res['RMSE']=dic_rmse
    res['MaxAE']=dic_mae
    with open(pathtarget,'w',encoding='utf8') as f:
        f.write(json.dumps(res))

tojson('path\\to\\SISSO.out','path\\to\\SISSO_out.json')