import numpy as np


def my_r2_score(y, pre, float_num):
    average_y = np.mean(y)
    tot = sum([(y[i] - average_y) ** 2 for i in range(len(y))])
    res = 0
    for i in range(len(y)):
        if y[i] - float_num[i] <= pre[i] <= y[i] + float_num[i]:
            res += 0
        elif y[i] - float_num[i] > pre[i]:
            res += (y[i] - float_num[i] - pre[i]) ** 2
        else:
            res += (y[i] + float_num[i] - pre[i]) ** 2
    return (1 - res/tot)

a = my_r2_score([191.13,197.17,189.11], [160,180,200], [28,20,30])

print(a)



