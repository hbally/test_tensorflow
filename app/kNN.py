# -*- coding: utf-8 -*-
# @Time    : 12/21/17 2:28 PM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : kNN.py
# @Software: PyCharm

from numpy import  *
import operator

def createDataSet():
    #样本数据集
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    #样本分类集
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    KNN算法 ：k-近邻算法简单实现
    :param inX:用于输入的分类向量
    :param dataSet:输入样本集
    :param labels:标签向量集即结果
    :param k:用于选择最近邻近的数据 即KNN中的K的由来
    :return:输出返回的分类结果
    '''
    # shape 查看矩阵或者数组的维数 0表示查看第一维的长度
    dataSetSize = dataSet.shape[0]
    # 计算与训练集每个特征值的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #矩阵每个元素上平方
    sqDiffMat = diffMat**2
    #axis=1 矩阵行求和
    sqDistances = sqDiffMat.sum(axis=1)
    #开方求出两点之间的--距离
    distances = sqDistances**0.5
    # 按距离求大小排序，距离越小就表示最相似，即最相邻
    sortedDistIndicies = distances.argsort()#数组值从小到大的索引值
    #计算前K个相似的数据集中 最多的分类
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #降序排列 即距离最小的在最上面
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    group,labels = createDataSet()
    result = classify0([0,0],group,labels,3)
    print(result)
    # B 最后KNN算法推测出结果分类维B
