# -*- coding: utf-8 -*-
# @Time    : 12/28/17 9:36 AM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : logRegres.py
# @Software: PyCharm
# Logistic 回归梯度上升优化算法

from numpy import *


def loadDataSet():
    '''测试的数据集'''
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 将特征值x0都设置为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    '''sigmoid 阶跃函数'''
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''梯度上次算法：该处是一次性处理数据即批处理'''
    # 样本数据矩阵 x0,x1,x2三个特征值，x0=1.0
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    # transpose()将向量转置，classlabels行向量变列向量
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix)  # m行，n列的样本数据
    # alpha是向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 假设权重都为1，
    weights = ones((n, 1))
    # 然后是循环迭代，算出最佳回归参数
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult  #该处需要计算300次乘法
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    '''随机梯度上升算法'''
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''对随机梯度上升算法进行优化'''
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def run_gradAscent():
    dataArr, labelMat = loadDataSet()
    weigths = gradAscent(dataArr, labelMat)
    # print("  weigths %s" % (weigths))
    #plotBestFit
    #getA 矩阵转变为数组ndarray
    plotBestFit(weigths.getA())
    ##随机梯度上升算法,并不是最佳拟合函数
    weigths = stocGradAscent0(array(dataArr), labelMat)
    print("  weigths %s" % (weigths))
    # plotBestFit
    # getA 矩阵转变为数组ndarray
    plotBestFit(weigths)
    #
    weigths = stocGradAscent1(array(dataArr), labelMat,1500)
    print("  weigths %s" % (weigths))
    # plotBestFit
    # getA 矩阵转变为数组ndarray
    plotBestFit(weigths)

def plotBestFit(weights):
    '''划出logistic回归最佳拟合直线的函数'''
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #x和y是怎么个意思？？
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    # print("x shape :%s " % x)
    # print("y shape :%s " % y)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == "__main__":
    # run_gradAscent()
    multiTest()

    print("ones: %s" % ones((3,4)))
