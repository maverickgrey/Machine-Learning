import random
import numpy as np 

class LinearRegression:
    def __init__(self,train_X,Y,features,norm_value,type='simple',bias=True):
        self.norm_value=None
        if type == 'ridge' or type == 'lasso':
            self.norm_value = norm_value

        self.type = type
        self.bias = bias

        if not bias:
            self.train_X = np.array(train_X,dtype=np.float)
        else:
            ones = np.ones((len(Y),1))
            self.train_X = np.append(ones,train_X,axis=1)
        self.Y = np.array(Y,dtype=np.float)
        self.features = features

        if not self.bias:
            self.w = np.random.random(features)
        else:
            # 指定w的第一维为b
            self.w = np.random.random(features+1)
                

    # 损失函数——最小二乘:pred是预测值数组，pred是标签数组
    def MSE_loss(self,pred,label):
        diff = (pred-label)
        loss = np.dot(diff.T,diff)/len(label)
        return loss
    
    # 对wi在当前点求关于最小二乘的梯度值
    def simple_gradient(self):
        grad = []
        if self.bias:
            partial_b = (self.pred(self.train_X)-self.Y).sum()
            grad.append(partial_b)
            for j in range(1,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = self.pred(self.train_X[i])-self.Y[i]
                    diff *= self.train_X[i][j]
                    partial_w += diff
                #partial_w /= len(self.Y)
                grad.append(partial_w)
        else:
            for j in range(0,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = np.dot(self.w,self.train_X[i])-self.Y[i]
                    diff *= self.train_X[i][j]
                    partial_w += diff
                #partial_w /= len(self.Y)
                grad.append(partial_w)
        return np.array(grad)
    
    # 对于ridge回归求loss的梯度值
    def ridge_gradient(self,lamb):
        grad = []
        if self.bias:
            partial_b = (self.pred(self.train_X)-self.Y).sum()
            grad.append(partial_b)
            for j in range(1,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = self.pred(self.train_X[i])-self.Y[i]
                    diff *= self.train_X[i][j]
                    partial_w += diff
                partial_w += lamb*self.w[j]
                #partial_w /= len(self.Y)
                grad.append(partial_w)
        else:
            for j in range(0,len(self.w)):
                partial_w = 0
                for i in range(len(self.Y)):
                    diff = np.dot(self.w,self.train_X[i])-self.Y[i]
                    diff *= self.train_X[i][j]
                    partial_w += diff
                partial_w += lamb*self.w[j]
                #partial_w /= len(self.Y)
                grad.append(partial_w)
        return np.array(grad)

    def quick_gradient(self,ridge,norm):
        XX = np.matmul(self.train_X.T,self.train_X)
        grad = np.matmul(XX,self.w)
        grad -= np.matmul(self.train_X.T,self.Y)
        if ridge==True:
            grad += norm * self.w
        if self.bias:
            partial_b = (self.pred(self.train_X)-self.Y).sum()
            grad[0] = partial_b
        return grad

    # 梯度下降法，lambda为学习率
    def gradient_descent(self,lr,t_step=1000,threshold = 1e-7):
        step = 0
        steps = []
        losses = []
        grad = self.gradient()
        # 终止条件：当达到指定迭代步数或者梯度已经很小时
        while(step<t_step) and (np.dot(grad,grad)>threshold):
            pred = self.pred(self.train_X)
            loss = self.MSE_loss(pred,self.Y)
            losses.append(loss)
            steps.append(step)
            self.w -= lr * grad
            grad = self.gradient()
            step += 1
        return steps,losses

    # 预测值
    def pred(self,X):
        return np.matmul(X,self.w)


    # 根据回归类型返回对应的梯度    
    def gradient(self):
        if self.type == 'simple':
            return self.quick_gradient(ridge=False,norm=self.norm_value)
        elif self.type == 'ridge':
            return self.quick_gradient(ridge=True,norm=self.norm_value)


# X=[[1,3]]
# Y=[1]
# r = LinearRegression(X,Y,2,True)
# print(r.X)
# print(r.w)
# print(r.gradient())
# r.gradient_descent(0.001)
# print(r.pred([1,1,3]))

