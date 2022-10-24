# 该文件用于读取数据等
import numpy as np
import pandas as pd
import math

train_ratio = 0.7

# 返回训练集
def read_training(path='./data/housing.csv'):
    train_data=[]
    train_label = []
    mean = []
    var = []

    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Boston = pd.read_csv(path,delimiter=r"\s+",names=column_names)
    train_num = int(Boston.shape[0]*train_ratio)
    Boston = Boston[:train_num]
    #求每一列的均值和方差以进行归一化
    for col in column_names:
        mean.append(Boston.loc[:,col].mean())
        var.append(Boston.loc[:,col].var())
    for row in range(train_num):
        example = np.array(Boston.iloc[row])
        label = example[-1]
        example = np.delete(example,-1)
        train_data.append(example)
        train_label.append(label)
    return train_data,train_label,mean,var

# 返回测试集
def read_test(path = './data/housing.csv'):
    test_data=[]
    test_label = []
    mean = []
    var = []

    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Boston = pd.read_csv(path,delimiter=r"\s+",names = column_names)
    train_num = int(Boston.shape[0]*(train_ratio))
    test_num = Boston.shape[0]-train_num
    Boston = Boston[train_num:]
    #求每一列的均值和方差以进行归一化
    for col in column_names:
        mean.append(Boston.loc[:,col].mean())
        var.append(Boston.loc[:,col].var())
    for row in range(test_num):
        example = np.array(Boston.iloc[row])
        label = example[-1]
        example = np.delete(example,-1)
        test_data.append(example)
        test_label.append(label)
    return test_data,test_label,mean,var


# 将输入的X进行归一化，而不对Y进行归一化
def norm(train,mean,var):
    train_res = train
    for j in range(len(mean)-1):
        for i in range(len(train)):
            train_res[i][j] -= mean[j]
            train_res[i][j] /= math.sqrt(var[j])
    return train_res

def stat():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Boston = pd.read_csv("./data/housing.csv",delimiter=r"\s+",names=column_names)
    print(Boston.corr())


# test_data,test_label,mean,var=read_test()
# test_data = norm(test_data,mean,var)
# print(test_data)
stat()