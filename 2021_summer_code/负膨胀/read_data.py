import pandas as pd
# 显示所有的列
pd.set_option('display.max_columns', None)

# 显示所有的行
pd.set_option('display.max_rows', None)


# def read_csv():
#     raw = pd.read_csv('filter_data/result.csv', encoding='gbk')
#     raw.fillna(0, inplace=True)
#     raw.drop(columns=['formula'], inplace=True)
#     raw['alpha'] = - raw['alpha']
#     raw.astype('float64')
#     values = raw.values
#     test, train = dict(), dict()
#     test['feature'] = values[0:3, 0:90]
#     test['target'] = values[0:3, 90:94]
#     train['feature'] = values[3:, 0:90]
#     train['target'] = values[3:, 90:94]
#     return train, test


# def read_csv():
#     raw = pd.read_csv('filter_data/data.csv', encoding='gbk')
#     raw.fillna(0, inplace=True)
#     raw.drop(columns=['formula'], inplace=True)
#     raw['neg_inflation'] = - raw['neg_inflation']
#     raw.astype('float64')
#     values = raw.values
#     test, train = dict(), dict()
#     train['feature'] = values[3:, 4:15]
#     train['target'] = values[3:, 0:4]
#     test['feature'] = values[0:3, 4:15]
#     test['target'] = values[0:3, 0:4]
#     return train, test


def read_csv():
    raw = pd.read_csv('filter_data/data4.csv', encoding='gbk')
    raw.fillna(0, inplace=True)
    raw.drop(columns=['formula'], inplace=True)
    raw['neg_inflation'] = - raw['neg_inflation']
    raw.astype('float64')
    values = raw.values
    test, train = dict(), dict()
    train['feature'] = values[3:, 0:15]
    train['target'] = values[3:, 15:19]
    test['feature'] = values[0:3, 0:15]
    test['target'] = values[0:3, 15:19]
    return train, test


if __name__ == '__main__':
    train, test = read_csv()
    print(test)
