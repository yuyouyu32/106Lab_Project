import re

filename1 = 'SISSO.in_model'
filename2 = 'parameters.txt'
filename3='SISSO.in'

def match_list(filename2):
    with open(filename2, "r+") as f2:  # 设置文件对象
        list1 = []
        list2 = []
        alllines = f2.readlines()
        del alllines[-1]
        del alllines[-1]
        for i in alllines:
            mark1 = re.match('\w*(?=:)', i)
            if mark1:
                list1.append(mark1.group())
            mark2 = re.search(':[\S]+', i)
            if mark2:
                str2 = mark2.group()
                str2=re.sub(':', '=', str2,1)
                list2.append(str2)
        for i in list2:
            if (i[0] == ':'):
                i.lstrip(':')
    return list1, list2


if __name__ == '__main__':
    list1, list2 = match_list(filename2)
    lines1 = []
    with open(filename1, "r+") as f1:  # 设置文件对象
        for i in f1.readlines():
            line1 = []
            for j in range(len(list1)):
                mark1=re.match(list1[j],i)
                if mark1:
                  i=re.sub('=[\S]+',list2[j],i,1)
            line1.append(i)
            lines1.extend(line1)
        f1.close()
    f2=open(filename3,'a')
    for i in lines1:
        f2.write(i)
    f2.close()




