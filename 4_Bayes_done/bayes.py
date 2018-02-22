'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

# 输入数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def print_lol(the_list):
	for each_flick in the_list:
		if isinstance(each_flick,list):
			print_lol(each_flick)
		else:
			print(each_flick)
            
   
    
                 
# 5 采用set结构获取词汇并集
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet: #列表循环
        #vocabSet = vocabSet | set(document) #union of the two sets
        vocabSet = vocabSet.union(set(document)) #取并集，并赋新值，重复项只保留一个
    return list(vocabSet) #转换set 为 list

# 参数1位词汇表；参数2位一个句子；返回参数1长度的对应单词是否出现的标记
def setOfWords2Vec(vocabList, inputSet):
    l = len(vocabList) #list类型长度
    returnVec = [0]*len(vocabList) #词汇表设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #print(vocabList.index(word)) 指定位置
        else: 
            print("the word: {0} is not in my Vocabulary!",format(word))
    # print(returnVec) 
    return returnVec


# 555 最重要的Bayes分类函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #文档数量 6
    numWords = len(trainMatrix[0]) #文词汇表长度 32
    pAbusive = sum(trainCategory)/float(numTrainDocs) #侮辱性占比
    
    p0Num = ones(numWords); p1Num = ones(numWords) #numpy的函数,返回narray     #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    #p0Num: 分子; p0Denom:分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #如果文档i是侮辱性
            p1Num += trainMatrix[i] #分子为一个array;两个array相加
            p1Denom += sum(trainMatrix[i]) #分母为一个总值；两个数字相加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #print(p1Num,p1Denom)
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

# 555 获取最大后验概率 从而做出判断
# vec2Classify：array(是否出现32长度); p0Vec: 32长度
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # print(vec2Classify*p1Vec)
    # vec2Classify * p1Vec : array相乘 后求和为一固定值
    # LOG相加就是相乘，得出贝叶斯分类的公式分子进行对比
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) # = 似然*先验概率
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # 构建setOfWords2Vec组建的list的list集合
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # 下面进行测试1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    # 测试2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# 参数1位词汇表；参数2位一个句子；返回参数1长度的对应单词是否出现的标记
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #词汇表设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #区别在=1 or +=1 
    return returnVec

# 正则表达式切分词汇 使用package:ref
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString) #split字符串bigString,返回list
    #直接用python的字符串的split函数:效果不好,包含逗号等符号
#    print(bigString.split()) #
    #借用re包中的split函数,需要正则表达
#    print(re.split(r'\W*', bigString))
    
    #tok为string字符串,lower()表示全部转换为小写
#    ree = []
#    for tok in listOfTokens:
#        if len(tok) > 2:
#            ree.append(tok)
#    print(ree)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    # docList包含所有文本单词
    # classList为每个文本的类别0或1
    # fullText为所有文本累积的单词
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        # wordList每次更新,包含本次文本单词
        wordList = textParse(open('email\spam\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
  
    #555 调用函数创建词汇表(排除重复)
    vocabList = createVocabList(docList)#create vocabulary
    
    trainingSet = list(range(50)); testSet=[]  #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    #获取整数型的训练集及其分类标志
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet: #train the classifier (get probs) trainNB0
        #print(setOfWords2Vec(vocabList, docList[docIndex])) #参数2:具体句子 bagOfWords2VecMN
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])


    #训练
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #如果分类错误,即不等于给定的分类值
            errorCount += 1 #分错加1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,fullText


if __name__ == '__main__':
    spamTest()
    

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
        
