'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

# 便利函数: open 读取文件
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("d:/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip() #去除前后空格
        lineArr = line.strip().split() #系统自带的split函数,切割为列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #没有float可能为字符串
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# 定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 555:计算梯度并返回最优系数w
def gradAscent(dataMatIn, classLabels):
    dataMatrix = asmatrix(dataMatIn)             #convert to NumPy matrix asmatrix = mat
    labelMat = asmatrix(classLabels).transpose() #convert to NumPy matrix
    #print(labelMat)
    m,n = shape(dataMatrix) #矩阵的行m=100,矩阵列:n=3
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))    #初始系数值为1,1,1. 按梯度方向走步长为0.001; 方位为gradient
    for k in range(maxCycles):              #heavy on matrix operations
        # 系数权重weights*对应x数据dataMatrix,通过sigmoid函数计算获取y值
        h = sigmoid(dataMatrix*weights)     #matrix mult dataMatrix:100*3; weights:3*1; 结果:100*1
        # y值与已知分类差异为error,再乘以原始x数据,即为梯度方向 555 此次需理论支持
        error = (labelMat - h)              #vector subtraction error:100*1
        gradient = dataMatrix.transpose()* error # [3*100] * [100*1] = [3*1] gradient
        weights = weights + alpha * gradient #matrix mult
    return weights

# 便利函数:画图
def plotBestFit(weights):
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
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 55: 随机梯度上升算法0
# 不是循环500次,而是100次,即每个样本一次
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        gradient = error * dataMatrix[i] # error是一个float64，而dataMatrix[i]是一个列表;此处已转换为ndarray
        weights = weights + alpha * gradient
    return weights

# 555: 随机梯度上升算法1
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m)) # range类型修改为list类型
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #注意常数项.保证alpha不断变小，但不为0. 随着迭代次数增加, 步长alpha逐步减小
            #print(alpha)
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant #改动2：随机选取
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

if __name__=='__main__':
    dataSet, labelMat = loadDataSet()
    #wei = gradAscent(dataSet,labelMat);print(wei)
    #wei0 = stocGradAscent0(asarray(dataSet),asarray(labelMat));print(wei0)
    wei1 = stocGradAscent1(asarray(dataSet),asarray(labelMat));print(wei1)
    plotBestFit(wei1)  #getA():将matrix转为ndarray
    #help(asmatrix)

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
        
