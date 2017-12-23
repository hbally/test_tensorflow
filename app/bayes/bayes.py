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


def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯算法
    :param trainMatrix:文档矩阵
    :param trainCategory:每篇类别标签所构成的向量
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


if __name__ == "__main__":
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[1]))
    # [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
