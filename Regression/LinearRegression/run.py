from utils import read_data_as_vec,norm
from regression import LinearRegression
import numpy as np

train,labels,mean,var = read_data_as_vec()
train_n,labels_n = norm(train,labels,mean,var)

r = LinearRegression(train_n,labels_n,13,True)
print(r.w)
print(np.dot(r.gradient(),r.gradient()))
r.gradient_descent(lr=0.001)
print(np.dot(r.gradient(),r.gradient()))
