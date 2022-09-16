import random
import math
import numpy as np


class Chromosome(object):
    def __init__(self, length=6, bit=3, p_mutate=0.1, p_crossover=0.5, cross_pos=3):
        self.chromosome = []
        # 存放染色体的数组
        self.p_mutate = p_mutate
        # 基因变异概率
        self.p_crossover = p_crossover
        # 交叉的概率
        self.length = length
        # 染色体长度
        self.bit = bit
        # 一个数用多少位基因表示
        self.cross_pos = cross_pos
        # 基因交叉点，从该点往后两个染色体进行互换

        self.generate()

    def mutate(self):
        rand = random.randint(1, 1/self.p_mutate)
        if rand == 1:
            print("变异了！")
            rand2 = random.randint(0, self.length-1)
            if self.chromosome[rand2] == 0:
                self.chromosome[rand2] = 1
            else:
                self.chromosome[rand2] = 0

    def cross_over(self, chromosome2):
        rand = random.random()
        if rand < self.p_crossover:
            for i in range(self.cross_pos, self.length):
                self.chromosome[i], chromosome2.chromosome[i] = chromosome2.chromosome[i], self.chromosome[i]

    def decode(self):
        x_variety = []
        i = 0 
        while(i < self.length):
            x = 0
            j = 1
            while(j <= self.bit):
                x += self.chromosome[i+self.bit-j]*math.pow(2, j-1)
                j += 1
            i += self.bit
            x_variety.append(x)
        return x_variety 

    def generate(self):
        for i in range(self.length):
            self.chromosome.append(random.randint(0,1))

    def get_chromosome(self):
        return self.chromosome

    def show(self):
        print(self.chromosome)


class GA(object):
    def __init__(self, size, times):
        self.population = []                      
        # 种群列表，存放每个个体的染色体

        self.adaptability = []
        # 适应度列表，存放每个个体的适应度

        self.selection = []
        # 选择概率列表，存放每个个体被选中的概率

        self.children = []
        # 子代列表，存放子代个体的染色体

        self.size = size
        # 种群个体数目

        self.times = times
        # 迭代次数

        self.pop_generate()

    # 适应度函数
    def function(self, x):
        return x[0]*x[0] + x[1]*x[1]

    # 生成一个个体数位size的随机种群，并初始化其他参数
    def pop_generate(self):
        for i in range(0,self.size):
            ch = Chromosome()
            self.population.append(ch)
            self.selection.append(0)
            self.adaptability.append(0)

    # 获得当前种群信息
    def get_pop(self):
        pop = []
        for i in range(0,self.size):
            pop.append(self.population[i].get_chromosome())
        return pop

    # 显示当前种群个体的基因型
    def show_pop(self):
        print("该种群:",self.get_pop())

    # 查看子代的染色体
    def show_children(self):
        children = []
        try:
            for i in range(0,self.size):
                self.children[i].show()
        except Exception:
            print("No child!")

    # 计算当前种群的适应度adaptability
    def compute_adpt(self):
        for i in range(0, self.size):
            x = self.population[i].decode()
            self.adaptability[i] = self.function(x)
        print("当前种群适应度为：", self.adaptability)

    # 计算子代适应度
    def compute_adptc(self):
        for i in range(0, self.size):
            x = self.children[i].decode()
            self.adaptability[i] = self.function(x)
        print("当前子代适应度为：", self.adaptability)


    # 计算每个个体被选择的概率selection
    def compute_sel(self):
        sum = 0
        for i in range(self.size):
            sum += self.adaptability[i]
        for j in range(self.size):
            self.selection[j] = (self.adaptability[j]/sum)
        print("当前各个体被选择的概率：", self.selection)    

    # 对亲代进行自然选择
    def select_parent(self):
        # 轮盘赌算法选取亲代
        parents = []
        i = 0
        while i < 2:
            random.seed()
            rand = random.random()
            accum = 0
            selec = -1
            for j in range(0, self.size):
                accum += self.selection[j]
                if accum > rand:
                    selec = j
                    break
            parents.append(self.population[selec])
            i += 1
        return parents

    # 用亲代生成子代,在交叉互换后生成选取适应度较大的加入后代池中
    def generate_child(self):
        children = []
        parents = []
        i = 0
        while i < self.size:
            parents = self.select_parent()
            parents[0].cross_over(parents[1])
            adpt0 = self.function(parents[0].decode())
            adpt1 = self.function(parents[1].decode())
            if adpt0 > adpt1:
                children.append(parents[0])
            else:
                children.append(parents[1])
            i += 1
        return children 

    # 获取精英子代
    def get_elite(self):
        max_adpt = max(self.adaptability)
        for i in range(len(self.adaptability)):
            if self.adaptability[i] == max_adpt:
                print("该次进化的最佳个体适应度为：",self.adaptability[i])
                break

    def run(self):
        index = 0
        while index < self.times:
            self.compute_adpt()
            self.compute_sel()
            self.children = self.generate_child()
            self.compute_adptc()
            self.get_elite()
            self.population = self.children
            index += 1


if __name__ == "__main__":
    ga = GA(10, 50)
    ga.run()