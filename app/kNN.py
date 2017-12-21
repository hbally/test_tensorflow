# -*- coding: utf-8 -*-
# @Time    : 12/21/17 2:28 PM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : kNN.py
# @Software: PyCharm

from numpy import *
import operator


def createDataSet():
    # 样本数据集
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 样本分类集
    labels = ['A', 'A', 'B', 'B']
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
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 矩阵每个元素上平方
    sqDiffMat = diffMat ** 2
    # axis=1 矩阵行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方求出两点之间的--距离
    distances = sqDistances ** 0.5
    # 按距离求大小排序，距离越小就表示最相似，即最相邻
    sortedDistIndicies = distances.argsort()  # 数组值从小到大的索引值
    # 计算前K个相似的数据集中 最多的分类
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 降序排列 即距离最小的在最上面
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def run_classify0():
    '''执行分类器classify0'''
    group, labels = createDataSet()
    result = classify0([0, 0], group, labels, 3)
    print(result)
    # B 最后KNN算法推测出结果分类维B


##############################################################################

def file2matrix(filename):
    '''预处理数据：将数据源file转为矩阵'''
    # 心动等级3，2,1
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # get the number of lines in the file
    # 创建0填冲的numpy矩阵，但是numpy是二位矩阵，这里设置为3
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    index = 0
    # 遍历每行的数据
    for line in arrayOLines:
        line = line.strip()  # 截取所有的回车
        listFromLine = line.split('\t')  # 按|t分割数据list集合
        returnMat[index, :] = listFromLine[0:3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def run_file2matrix():
    datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    # print(" datingDataMat %s" % datingDataMat )
    # print(" datingLables %s" % mat(15.0*array(datingLables)))
    # 看矩阵不能够直观的感受到数据类别，需要图像化才好
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 只取了第二和第三列的数据 所有的都彼此包含其中很难得出结论
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLables),15.0*array(datingLables))
    # 只取了第一和第二列的数据，相对上面的图像，该图像轮廓清晰，有更好的效果显示分类区域
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLables), 15.0 * array(datingLables))
    plt.show()

def autoNorm(dataSet):
    '''归一化特征值，让所有的特征值的取值范围在【0-1】或者【-1,1】
       转化0到1区间值：
       newvalue = (oldValue - min)/(max - min )
    '''
    minVals = dataSet.min(0)#min(0)从列中选取最小值，[ 0.        0.        0.001156]
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #[  9.12730000e+04   2.09193490e+01   1.69436100e+00]
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def run_autoNorm():
    datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(" normDataSet %s" % normDataSet )
    print(" ranges %s" % ranges )
    print(" minVals %s" % minVals )


def datingClassTest():
    '''1.取样本的前100个数据做测试
       2.txt源数据的后900个数据作为样本数据
       3.推测结果和实际结果比较，计算出出错率
    '''
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    #对原始数据归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #计算测试向量的数量确定那些数据用于测试
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0#记录推测错误的数量
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

def classifyPerson():
    '''简单的一个交互工具：
       输入特征值，内部通过分类器预测出类型
    '''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])


if __name__ == "__main__":
    # run_autoNorm()
    # datingClassTest()
    classifyPerson()