#决策树回归

# 决策树的叶子节点
class Node:
    def __init__(self,left,right,X,Y,features):
        # 当前结点的分界值，如果该结点是叶子结点的话则值为最终预测值（均值）
        self.value = 0
        # 左右结点指针
        self.left = left
        self.right = right
        # 当前结点的值是描述哪个特征的(特征的编号)
        self.dim = 0
        # 在当前结点还需要处理的数据
        self.X = X
        self.Y =Y
        # 描述当前结点包含的数据有多少个维度
        self.features = features
        
    # 对于指定维度的数据，找到一个当前最好的划分点.fn为指定的特征编号、step为步长
    def find_division(self,fn,step):
        feature_data = self.X[fn]
        x_max = 0xffffffff
        x_min = 0x8fffffff
        for key in feature_data:
            if key > x_max:
                x_max = key
            if key< x_min:
                x_min = key
        division = x_min+step
        optimal = division
        min_ssrs = 0x8fffffff
        while(division<=x_max):
            #比划分点小的元素
            left_data = {}
            left_sum = 0
            left_avg=0
            #比划分点大或等的元素
            right_data = {}
            right_sum = 0
            right_avg = 0
            for key in feature_data:
                if key < division:
                    left_data[key] = feature_data[key]
                    left_sum += self.Y[feature_data[key]]
                else:
                    right_data[key] = feature_data[key]
                    right_sum += self.Y[feature_data[key]]
            left_avg = left_sum/len(left_data)
            right_avg = right_sum/len(right_data)
            # 计算Ssrs
            ssrs = self.ssrs(left_data,left_avg,right_data,right_avg)
            if ssrs<min_ssrs:
                optimal = division
                min_ssrs = ssrs
            division += step    
        return optimal,min_ssrs

    def ssrs(self,left_data,left_avg,right_data,right_avg):
        ssrs = 0
        for v in left_data:
            srrs += (self.Y[left_data[v]]-left_avg)**2
        for v in right_data:
            srrs += (self.Y[right_data[v]]-right_avg)**2
        return ssrs
    
    # 根据当前结点的数据获得分界点及左右结点
    def split(self):
        min_ssrs = 0x8fffffff
        dim = 0
        div = 0
        for fn in range(self.features):
            division,ssrs = self.find_division(fn,1)
            if ssrs < min_ssrs:
                dim = fn
                div = division
        self.dim = dim
        self.value = div
        left_data,right_data = self.get_child_data(self.dim,self.value)
        self.left = Node(None,None,left_data,self.Y,self.features)
        self.right = Node(None,None,right_data,self.Y,self.features)

    # 获得当前结点应当向左右结点应当传的数据
    def get_child_data(self,fn,div):
        left_data = []
        right_data = []
        for _fn in self.features:
            if _fn != fn:
                left_data.append(self.X[_fn])
                right_data.append(self.X[_fn])
            else:
                left_temp = []
                right_temp = []
                for v in self.X[_fn]:
                    if v<div:
                        left_temp[v]=self.X[_fn][v]
                    else:
                        right_temp[v] = self.X[_fn][v]
                left_data.append(left_temp)
                right_data.append(right_temp)
        return left_data,right_data
            

class RegressionTree:
    def __init__(self,features,X,Y):
        self.features = features
        self.real_roots = Node(0,None,None,self.features,)
        self.Y = Y
        self.X = self.process_X(X)
       
    # 给出以当前结点出发应该找到的叶子结点
    def find_leaf(self,node,variant):
        result = node
        while(not self.is_leaf(result)):
            result = self.find_next(result,variant)
        return result
    
    # 访问当前结点，给出下一个应该访问的结点
    def find_next(self,node,variant):
        d_feature = node.d_feature
        if self.is_leaf(node):
            return node
        else:
            if variant[d_feature] < node.value:
                return node.left
            else:
                return node.right

    # 判断当前结点是否是叶子结点
    def is_leaf(self,node):
        return node.left == None and node.right == None
    
    # 给定特征向量，返回预测值
    def predict(self,variant):
        leaf = self.find_leaf(self.real_roots,variant)
        return leaf.value
    
    #将训练数据处理成按特征列存储，每个值还有一个偏移量，用来指向Y
    def process_X(self,X):
        data = []
        for fn in range(self.features):
            fn_data = {}
            for row in range(len(X)):
                fn_data[X[row][fn]] = row
            data.append(fn_data)
        return data
    

