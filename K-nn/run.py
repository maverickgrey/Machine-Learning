import knn
import utils

train_img_path = "./train-images.idx3-ubyte"
test_img_path = './t10k-images.idx3-ubyte'
train_label_path = './train-labels.idx1-ubyte'
test_label_path = './t10k-labels.idx1-ubyte'
K=3

def run():
    print("正在为knn装载数据库...")
    train_pics,train_labels = utils.get_database(train_img_path,train_label_path)
    train_pics = knn.get_mean(train_pics)
    print("装载完毕!")
    res = knn.predict(test_img_path,train_pics,train_labels,K)
    test_labels = utils.load_labels(test_label_path)
    acc = knn.acc(res,test_labels)
    print("the accuracy:{}".format(acc))

if __name__ == '__main__':
    run()