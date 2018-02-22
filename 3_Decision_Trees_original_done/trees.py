'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''

from math import log
import operator

# 获取数据集
    def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算数据集的总体熵 shannonEnt 熵一定为正数;分类越多,越混杂,熵越高.
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} #dict类型
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]  #currentLabel:字典的key: List的倒数第一个是类别的别致
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
        #print(shannonEnt)
    return shannonEnt

# 对数据集针对特征axis进行数据划分
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
  
  # 555 信息增益计算核心
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征数量 #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   #全部数据的熵
    bestInfoGain = 0.0; bestFeature = -1
    #i循环+value循环: 调用splitDataSet
    for i in range(numFeatures):        #range的用法??? #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # print(featList)
        uniqueVals = set(featList)       #get a set of unique values
        #print(uniqueVals)        
        newEntropy = 0.0
        # 55 计算每个特征i下的分类后的熵之和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #(list,int,set)
            prob = len(subDataSet)/float(len(dataSet)) #权重 |Dv| / |D|
            newEntropy += prob * calcShannonEnt(subDataSet)  #权重*熵    
        # 55 计算信息增加infoGain            
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # 获取最优信息增益减少对应的特征i
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


# 子函数:当叶子节点为1时使用
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 输出:以字典代替树的数据结构: 最终返回myTree (其实质为一个字典dict) 
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]    # 获取list: 类别

    # 下面两个return仅在递归时遇到
    a = classList[0]; b = classList.count(a);  c = len(classList);  d = len(dataSet[0])
    if classList.count(classList[0]) == len(classList): #如果所有类别都一样,即达到叶子节点
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet @如果只余下一个算例,也是叶子节点
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}; #print(myTree)
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    

if __name__=='__main__':
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet,labels)
    print(myTree)
    #print(dataSet)
    # print(chooseBestFeatureToSplit(dataSet))
    # dataSet[0][-1] = 'maybe' #在现有数据集的第一个列表中最后一个值更新数据 
    # calcShannonEnt(dataSet)
    



def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
