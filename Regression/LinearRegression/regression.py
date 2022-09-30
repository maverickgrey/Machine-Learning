from pyexpat import features
import random
import numpy as np 

class LinearRegression:
    def __init__(self,X,Y,features,bias=True):
        self.bias = bias
        if not bias:
            self.X = np.array(X,dtype=np.float)
        else:
            ones = np.ones((len(Y),1))
            self.X = np.append(ones,X,axis=1)
                
        
        self.Y = np.array(Y,dtype=np.float)
        self.features = features

        if not self.bias:
            self.w = np.random.random(features)
        else:
            # 指定w的第一维为b
            self.w = np.random.random(features+1)
            
    # 1、简单回归
    def simple_regression(self):
        pred = self.pred()
        loss = self.MSE_loss(pred,self.Y)
        

    # 损失函数——最小二乘
    def MSE_loss(self,pred,label):
        diff = (pred-label)
        print(diff)
        loss = np.dot(diff.T,diff)/len(label)
        return loss
    
    # 对wi在当前点求关于最小二乘的梯度值
    def gradient(self):
        grad = []
        if self.bias:
            partial_b = (self.pred(self.X)-self.Y).sum()/len(self.Y)
            grad.append(partial_b)
            for j in range(1,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = self.pred(self.X[i])-self.Y[i]
                    diff *= self.X[i][j]
                    partial_w += diff
                partial_w /= len(self.Y)
                grad.append(partial_w)
        else:
            for j in range(0,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = np.dot(self.w,self.X[i])-self.Y[i]
                    diff *= self.X[i][j]
                    partial_w += diff
                partial_w /= len(self.Y)
                grad.append(partial_w)
        return np.array(grad)

    # 梯度下降法，lambda为学习率
    def gradient_descent(self,lr,t_step=10000,threshold = 1e-7):
        step = 0
        grad = self.gradient()

        # 终止条件：当达到指定迭代步数或者梯度已经很小时
        while(step<t_step) and (np.dot(grad,grad)>threshold):
            self.w -= lr * grad
            grad = self.gradient()
            step += 1

    # 预测值
    def pred(self,X):
        return np.matmul(X,self.w)
        


# X=[[1,3]]
# Y=[1]
# r = LinearRegression(X,Y,2,True)
# print(r.X)
# print(r.w)
# print(r.gradient())
# r.gradient_descent(0.001)
# print(r.pred([1,1,3]))

