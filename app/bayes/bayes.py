# -*- coding: utf-8 -*-
# @Time    : 12/23/17 11:24 AM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : bayes.py
# @Software: PyCharm
from numpy import *


def loadDataSet():
    '''模拟造数据
        假设有六篇博客
    '''
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 人工分类了结果;1表示含有辱骂，0表示不含辱骂
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    '''创建词汇列表，用了set所以其中不会重复'''
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    文档词向量化
    :param vocabList: 词汇表
    :param inputSet: 输入某个文档
    :return:输出文档向量，文档中出现的单词在词汇表中出现则对应为1，没有出现则为0；
    '''
    # 首先创建与词汇表等长的全为0的集合
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    else:
        print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    '''同样是将文档向量化
       不同于setOfWords2Vec，bag集容许单词多次出现
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯算法
    :param trainMatrix:文档矩阵
    :param trainCategory:每篇类别标签所构成的向量
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    print("numTrainDocs = %s ,numWords = %s " % (numTrainDocs, numWords))
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 计算条件概率时 如果p(w 0 |1)p(w 1 |1)p(w 2 |1)中有某一项为0,那么求乘积的时候就 为了0
    # 所以，避免这个问题出现，即把所以单词的出现数初始化1，并将分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 太多的小数相乘就会非常小，取自然对数就可以避免四舍五入带来的错误，对数ln不会有任何损失；
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


def run_trainNB0():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    # [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    # 存储所有文档向量
    trainMat = []
    for postDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print("p0V = %s \n,p1V = %s \n ,pAb= %s  " % (p0V, p1V, pAb))


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):  # input is big string, #output is word list
    '''解析text'''
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = list(range(50));
    testSet = []  # create test set
    for i in range(10):
        # 随机的构建训练集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print(
                "classification error", docList[docIndex])
    # 输出错误率并且错误的文档
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


if __name__ == "__main__":
    # testingNB()
    spamTest()
