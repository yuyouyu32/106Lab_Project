import requests

url = 'http://sc.shu.edu.cn:7190'
# 上传的文件
files = {'file': open('job.tar.gz', 'rb')}         
# 携带的参数
data = {'uid':'shu','tid':'2019100801'}

r = requests.post(url, files = files, data = data)
print(r.text)
