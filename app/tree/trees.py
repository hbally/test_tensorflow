# -*- coding: utf-8 -*-
# @Time    : 12/22/17 11:11 AM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : trees.py 决策树构建
# @Software: PyCharm
from math import log
import operator


def calcShannonEnt(dataSet):
    '''计算数据集的信息量-熵'''
    # 计算数据集实例总数
    numEntries = len(dataSet)
    # 统计类别数量
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 通过熵计算公式计算
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def createDataSet():
    # 创建数据集
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def run_creatDataSet():
    # 创建数据集
    dataSet, labels = createDataSet()
    print(" dataSet = %s, labels = %s " % (dataSet, labels))
    # 计算信息量---熵
    print("shanonEnt: %s" % calcShannonEnt(dataSet))
    # shanonEnt: 0.9709505944546686
    # 增加分类，
    dataSet[0][-1] = 'maybe'
    print("shanonEnt: %s" % calcShannonEnt(dataSet))
    # shanonEnt: 1.3709505944546687
    # 可以看出越多的不确定性就代表越多的信息量，即熵就越大

    print(dataSet[0])


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet:待划分数据集
    :param axis:划分数据集的特征
    :param value:特征的返回值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            # extend表示集合合并到尾部
            reducedFeatVec.extend(featVec[axis + 1:])
            # append在集合尾部添加一项object
            retDataSet.append(reducedFeatVec)
    return retDataSet


def run_splitDataSet():
    # 创建数据集
    dataSet, labels = createDataSet()
    print(" dataSet = %s, labels = %s " % (dataSet, labels))
    splitData = splitDataSet(dataSet, 0, 1)  # 选择特征1，为1的数据集
    print("splitData = %s" % splitData)
    # splitData = [[1, 'yes'], [1, 'yes'], [0, 'no']]
    splitData = splitDataSet(dataSet, 0, 0)  # 选择特征1，为0的数据集
    print("splitData = %s" % splitData)
    # splitData = [[1, 'no'], [1, 'no']]


def chooseBestFeatureToSplit(dataSet):
    '''选择最好的分类-'''
    # 样本特征值长度 即多少列，因为最后项是分类结果所以减一
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    # 样本熵
    baseEntropy = calcShannonEnt(dataSet)
    # 存储最高信息增益
    bestInfoGain = 0.0;
    # 最佳特征值
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        # list comprehension 列表推导产生新的数据集合
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        # set数据集合和list所不同的是，每个值是互不相同的
        uniqueVals = set(featList)  # get a set of unique values
        # 新的信息熵
        newEntropy = 0.0
        # 计算每个划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 这里计算在子集合在总的集合中的百分比，
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益 老减去新 因为从无序变为有序不确定信息就会变少
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        # 将比较最大的信息增益
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            # 记录最佳特征
            bestFeature = i
    return bestFeature  # returns an integer


def run_chooseBestFeatureToSplit():
    # 创建数据集
    dataSet, labels = createDataSet()
    bestFeature = chooseBestFeatureToSplit(dataSet)
    print(" dataSet = %s, labels = %s " % (dataSet, labels))
    print(" bestFeature = %s" % (bestFeature))


def majorityCnt(classList):
    '''计算出出现最多的分类'''
    # classCount存储classList中每个类标签出现的频率
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # sorted字典排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''创建树--就是一个递归函数'''
    classList = [example[-1] for example in dataSet]
    # 递归函数的第一个终止条件：所有的标签完全相同，则直接返回该标签分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # 递归函数的第二个终止条件：没有任何特征项，只能返回出现最多的标签分类
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选出最好的特征项
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 特征项对应的名称
    bestFeatLabel = labels[bestFeat]
    # 构建出树结构存储在myTree
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 子特征标签
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        # 树干上开支，递归调用createTree
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def run_createTree():
    # 创建数据集
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)


def classify(inputTree, featLabels, testVec):
    '''
    决策树分类器
    :param inputTree: 决策树
    :param featLabels: 特征
    :param testVec: 测试向量
    :return:
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    '''存储决策树'''
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    '''读取决策树'''
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# import treePlotter


def class_lenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmaic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print("lensesTree :%s" % lensesTree)
    # treePlotter.createPlot(lensesTree)


if __name__ == "__main__":
#     # run_creatDataSet()
#     # run_splitDataSet()
#     # run_chooseBestFeatureToSplit()
#     # run_createTree()
    class_lenses()
