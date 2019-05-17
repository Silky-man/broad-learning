# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as scio
from BroadLearningSystem import BLS,BLS_AddEnhanceNodes,BLS_AddFeatureEnhanceNodes,bls_train_input,bls_train_inputenhance

def process(path):
    data = pd.read_csv(path, dtype='a')
    label = np.array(data['emotion'])
    img_data = np.array(data['pixels'])
    N_sample = label.size
    print(N_sample)
    # print label.size
    Face_data = np.zeros((N_sample, 56*56))
    Face_label = np.zeros((N_sample, 4), dtype=int)
    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ')
        # x_max = x.max()
        # x = x / (x_max + 0.0001)
        Face_data[i] = x
        Face_label[i, int(label[i])] = 1
        # if i < 10:
        #     print('i: %d \t ' % (i), Face_label[i])
    return Face_data,Face_label
#train_num = 163
#test_num = 50
traindata,trainlabel=process("/home/xgd/lijiawei/jaffe.csv")
testdata,testlabel=process("/home/xgd/lijiawei/jaffe1.csv")
# traindata = Face_data[0:train_num, :]
# trainlabel = Face_label[0:train_num, :]
# testdata = Face_data[train_num: train_num + test_num, :]
# testlabel = Face_label[train_num: train_num + test_num, :]
print("All is well")

N1 = 10  #  # of nodes belong to each window
N2 = 10  #  # of windows -------Feature mapping layer
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
print('-------------------BLS_ENHANCE------------------------')
BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)


'''
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Training Time
t0 = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
# BLS parameters
s = 0.8  #reduce coefficient
C = 2**(-30) #Regularization coefficient
N1 = 22  #Nodes for each feature mapping layer window 
N2 = 20  #Windows for feature mapping layer
N3 = 540 #Enhancement layer nodes
#  bls-网格搜索
for N1 in range(8,25,2):
    r1 = len(range(8,25,2))
    for N2 in range(10,21,2):
        r2 = len(range(10,21,2))
        for N3 in range(600,701,10):
            r3 = len(range(600,701,10))
            a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
            t0 += 1
            if a>t1:
                tt1 = N1
                tt2 = N2
                tt3 = N3
                t1 = a
            teA.append(a)
            tet.append(b)
            trA.append(c)
            trt.append(d)
            print('percent:' ,round(t0/(r1*r2*r3)*100,4),'%','The best result:', t1,'N1:',tt1,'N2:',tt2,'N3:',tt3)
meanTeACC = np.mean(teA)
meanTrTime = np.mean(trt)
maxTeACC = np.max(teA)   
'''
'''
#BLS随机种子搜索
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Train Time
t0 = 0
t = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
## BLS parameters
s = 0.8 #reduce coefficient
C = 2**(-30) #Regularization coefficient
#N1 = 10  #Nodes for each feature mapping layer window 
#N2 = 10  #Windows for feature mapping layer
#N3 = 500 #Enhancement layer nodes
dataFile = './/dataset//mnist.mat'
data = scio.loadmat(dataFile)
traindata,trainlabel,testdata,testlabel = np.double(data['train_x']/255),2*np.double(data['train_y'])-1,np.double(data['test_x']/255),2*np.double(data['test_y'])-1
u = 45
i = 0
L = 5
M = 50
for N1 in range(10,21,20):
    r1 = len(range(10,21,20))
    for N2 in range(12,21,10):
        r2 = len(range(12,21,10))
        for N3 in range(4000,4001,500):
            r3 = len(range(4000,4001,500))
            for i in range(-28,-27,2):
                r4 = len(range(-28,-27,2))
                C = 2**(i)
#                a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
                a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M)
#                t0 += 1
#                if a>t1:
#                    tt1 = N1
#                    tt2 = N2
#                    tt3 = N3
#                    t1 = a
#                    t = u
#                    i1 = i
#                tet.append(b)    
#                teA.append(a)               
#                trA.append(c)
#                trt.append(d)
#                print('NO.',t0,'total:',r1*r2*r3,'ACC:',a*100,'Pars:',N1,',',N2,',',N3,'C',i)
#                print('The best so far:', t1*100,'N1:',tt1,'N2:',tt2,'N3:',tt3,'C:',i1)
                print('working ...')
                print('teACC',teA,'teTime',tet,'trACC',trA,'trTime',trt)
'''
#Grid search for Regularization coefficient

#
'''
teA = list()
tet = list()
trA = list()
trt = list()
s = 0.8
#C = 2**(-30)
N1 = 10
N2 = 100
N3 = 8000
L = 5
M1 = 20
M2 = 20
M3 = 50
t0 = 0
t1 = 0
t2 = 0
for i in range(-30,-21,5):
    r1 = len(range(-30,-21,5))
    for u in range(10,50,1):
        r2 = len(range(10,50,1))
        C = 2**i
        t0 += 1 
#    a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)
        a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
        teA.append(a)
        tet.append(b)
        trA.append(c)
        trt.append(d)
        if a > t1:
            t1 = a
            t2 = i
            t = u
        print(t0,'percent:',round(t0/(r1*r2)*100,4),'%','The best result:', t1,'C',t2,'u:',t)
'''











