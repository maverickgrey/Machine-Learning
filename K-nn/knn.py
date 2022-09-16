import math
import numpy as np
from utils import *

# 计算两个向量的距离
def get_distance(vec_a,vec_b,type='euler'):
    if type == 'euler':
        diff = vec_a - vec_b
        distance = math.sqrt(np.sum(diff**2))
        return distance

# 对图像进行均一化
def get_mean(pics):
    # 将图像展开
    mean_pics = np.mean(pics,axis=0)
    pics = pics.astype(np.float)
    pics -= mean_pics
    return pics


# 返回给定向量周围K个最相邻的向量的标签
def search(vec,K,samples,labels):
    # 存储vec到所有其他样本的距离 
    distance = []
    res = []
    for sample in samples:
        dis = get_distance(vec,sample)
        distance.append(dis)
    script = np.argsort(distance)
    for i in range(len(script)):
        res.append(labels[script[i]])
    return res[:K]

# 在K个向量中统计各个标签的数量
def cnt_labels(labels):
    stat = {}
    # 临时保存当前出现最多的标签
    temp = labels[0]
    for label in labels:
        if label not in stat:
            stat[label] = 1
        else:
            stat[label] += 1
    for key in stat:
        if stat[key]>stat[temp]:
            temp = key
    return temp

#给定一张图片，用knn来确定它的标签
def knn(input,data,labels,K):
    k_neibors = search(input,K,data,labels)
    res = cnt_labels(k_neibors)
    return res

# 对测试集中的图片进行分类，返回每个图片的预测标签
def predict(test_path,database,labels,K):
    _,sample_cnt = get_head_info(test_path)
    t = open(test_path,'rb')
    results = []
    print("正在读取测试图片...")
    test = load_pics(test_path)
    test = get_mean(test)
    print("开始预测...")
    for offset in range(sample_cnt):
        if (offset+1) % 100 == 0:
            print("当前第{}张".format(offset+1))
        pic = test[offset]
        res = knn(pic,database,labels,K)
        results.append(res)
    t.close()
    return results

def acc(pred_labels,origin_labels):
    correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == origin_labels[i]:
            correct += 1
    return correct/len(pred_labels)