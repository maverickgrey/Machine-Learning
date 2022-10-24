from utils import read_training,norm,read_test
from regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def run():
    bias = True

    train_data,train_labels,train_mean,train_var = read_training()
    train_data = norm(train_data,train_mean,train_var)
    test_data,test_labels,test_mean,test_var = read_test()
    test_data = norm(test_data,test_mean,test_var)

    if bias:
        ones = np.ones((len(test_data),1))
        test_data = np.append(ones,test_data,axis=1)

    r = LinearRegression(train_data,train_labels,13,norm_value=0.5,type='ridge',bias=bias)
    print(r.gradient())
    print(np.dot(r.gradient(),r.gradient()))
    
    pred1 = r.pred(r.train_X)
    loss1 = r.MSE_loss(pred1,train_labels)
    print(loss1)

    steps,losses = r.gradient_descent(lr=0.0001)
    print(np.dot(r.gradient(),r.gradient()))
    plt.plot(steps,losses)
    plt.xlabel('times') 
    plt.ylabel('loss')
    plt.show()

    pred = r.pred(r.train_X)
    loss = r.MSE_loss(pred,train_labels)

    pred2 = r.pred(test_data)
    loss2 = r.MSE_loss(pred2,test_labels)
    print(loss)
    print(loss2)
    print(r.w)

if __name__ == '__main__':
    run()