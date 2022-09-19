import numpy as np


# head_length为数据头文件的大小（头文件包含了该数据集的一些基本信息，如每条数据的维度多少，每个维度的大小等）
# sample_count记录了该文件包含了多少条数据--训练集中有6万条数据
def get_head_info(filename):
    dimension = []
    with open(filename,'rb') as pf:
        data = pf.read(4)#获取magic number
        magic_num = int.from_bytes(data,byteorder='big')#bytes格式大尾端模式转换为int型
        dimension_cnt = magic_num & 0xff #获取dimension的长度,magic number最后一个字节
        for i in range(dimension_cnt):
            data = pf.read(4)  #获取dimension数据，dimension[0]表示图片的个数,图片文件中dimension[1][2]分别表示其行/列数值
            dms = int.from_bytes(data,byteorder='big')
            dimension.append(dms)
            
    sample_count = dimension[0]
    head_length = 4*len(dimension)+4
    return head_length ,sample_count


#mnist单个图片的大小
IMAGE_ROW = 28
IMAGE_COL = 28 

# 根据偏移量读取一张图片
# head_len为上个函数返回的head_length值
def read_image_p(pf,head_len,offset):
    image = np.zeros((IMAGE_ROW*IMAGE_COL),dtype=np.uint8)#创建空白数组存放图片，图片被拉成一维来存储
    pf.seek(head_len+IMAGE_ROW*IMAGE_COL*offset) #指向offset个图片的位置  
    for loc in range(IMAGE_ROW*IMAGE_COL):
        data = pf.read(1)#单个字节读
        pix = int.from_bytes(data,byteorder='big')#byte转为int
        image[loc] = pix
    return image


# 一次性读取全部的标签
def load_labels(filename):
    labels = []
    with open(filename,'rb') as l:
        data = l.read(4)
        magic = int.from_bytes(data,byteorder='big')
        data = l.read(4)
        cnt = int.from_bytes(data,byteorder='big')
        for offset in range(cnt):
            data = l.read(1)
            label = int.from_bytes(data,byteorder='big')
            labels.append(label)
    return labels

#  一次性读取进来全部的图片
def load_pics(filename):
    pics = []
    head_len,sample_cnt = get_head_info(filename)
    p = open(filename,'rb')
    for offset in range(sample_cnt):
        pic = read_image_p(p,head_len,offset)
        pics.append(pic)
    p.close()
    return np.array(pics)
        

def get_database(data_path,label_path):
    pics = load_pics(data_path)
    labels = load_labels(label_path)
    return pics,labels
