# 该文件用于读取数据等
import numpy as np
import pandas as pd
import math

# 把每行数据以列表的形式读入,返回输入特征X，标签Y，每一列的均值、方差
def read_data_as_vec(path="./data/housing.csv"):
    train = []
    labels = []
    mean = []
    var = []
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Boston = pd.read_csv("./data/housing.csv",delimiter=r"\s+",names=column_names)
    for col in Boston:
        avg = Boston[col].mean()
        sq = Boston[col].var()
        mean.append(avg)
        var.append(sq)

    for loc in range(Boston.shape[0]-500):
        data = np.array(Boston.loc[loc]).tolist()
        label = data[-1]
        data.pop(-1)
        train.append(data)
        labels.append(label)
    return np.array(train),np.array(labels),np.array(mean),np.array(var)

# 将输入X进行归一化的处理
def norm(train,labels,mean,var):
    train_res = train
    labels_res = labels
    for j in range(len(mean)-1):
        for i in range(len(train)):
            train_res[i][j] -= mean[j]
            train_res[i][j] /= math.sqrt(var[j])
    for i in range(len(labels)):
        labels_res[i] -= mean[-1]
        labels_res[i] /= math.sqrt(var[-1])
    return train_res,labels_res

train,labels,mean,var = read_data_as_vec()
train1,labels1 = norm(train,labels,mean,var)
