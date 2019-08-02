# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/11/23 13:05
"""
Convolutional Autoencoder.
"""
import os
import cv2
import numpy as np
from v2BroadLearningSystem import BLS,BLS_AddEnhanceNodes,BLS_AddFeatureEnhanceNodes,bls_train_input,bls_train_inputenhance
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,Input, UpSampling2D,Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import pandas as pd
np.random.seed(33)   # random seed，to reproduce results.

EPOCHS = 5
BATCH_SIZE =64

# input placeholder
input_image = Input(shape=(56, 56, 1))

# encoding layer
x = Conv2D(16, (3, 3), activation='relu', padding="same")(input_image)
x = MaxPool2D((2, 2), padding='same')(x)
#x = Conv2D(16,(2, 2),activation='relu',strides=2 )(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded= Conv2D(8,(2, 2), activation='relu',strides=2 )(x)
#x=  Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded =  MaxPool2D((2, 2), padding='same')(x)

# decoding layer
x = UpSampling2D((2, 2))(x)
#x=Lambda(lambda x: tf.image.resize_nearest_neighbor(x,(14,14)))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3),activation='sigmoid', padding='same')(x)

# build autoencoder, encoder, decoder
autoencoder = Model(inputs=input_image, outputs=decoded)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# training
# need return history, otherwise can not use history["acc"]


# Step1： load data  x_train: (60000, 28, 28), y_train: (60000,) x_test: (10000, 28, 28), y_test: (10000,)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
def aa(path,k):
    f=path
    fs = os.listdir(f)
    size=56
    #data = np.zeros([k, size*size*3], dtype=np.uint8)
    data = np.zeros([k, size,size], dtype=np.uint8)
    label = np.zeros([k], dtype=int)
    i = 0
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            img = cv2.imread(tmp_path,0)
            img = cv2.resize(img, (size,size))
            img_label = f1[:3]
            print(img_label)
            if img_label == 'bus':
                label[i] = 0
            elif img_label == 'fam':
                label[i] = 1
            elif img_label == 'fir':
                label[i] = 2
            elif img_label == 'hea':
                label[i] = 3
            elif img_label == 'jee':
                label[i] = 4
            elif img_label == 'min':
                label[i] = 5
            elif img_label == 'rac':
                label[i] = 6
            elif img_label == 'suv':
                label[i] = 7
            elif img_label == 'tax':
                label[i] = 8
            elif img_label == 'tru':
                label[i] = 9
            else:
                print("get label error.......\n")
            data[i] = img
            i = i + 1
    return data,label
x_train,y_train=aa("/home/xgd/图片/carmodel/train",1400)
x_test,y_test=aa("/home/xgd/图片/carmodel/test",200)
# Step2: normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Step3: reshape data, x_train: (60000, 28, 28, 1), x_test: (10000, 28, 28, 1), one row denotes one sample.
x_train = x_train.reshape((x_train.shape[0], 56, 56, 1))
x_test = x_test.reshape((x_test.shape[0], 56,56,1))

autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
# show images
#decode_images = autoencoder.predict(x_test)

encoder = Model(inputs=input_image, outputs=encoded)
train_images=encoder.predict(x_train)
_,a,b,c=train_images.shape
test_images=encoder.predict(x_test)
print(test_images.shape)
#################################################
def make(data11,label11,path):
    k=label11.size
    size=a
    data = np.zeros([k, size*size*c], dtype=np.uint8)
    label = np.zeros([k], dtype=int)
    for i in range(k):
        label[i] = label11[i]
        data[i][0:size * size*c] = np.ndarray.flatten(data11[i])

    with open("/home/xgd/lijiawei/"+path+".csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['emotion', 'pixels'])
        for i in range(len(label)):
            data_list = list(data[i])
            b = " ".join(str(x) for x in data_list)
            l = np.hstack([label[i], b])
            writer.writerow(l)
make(train_images,y_train,"num1")
make(test_images,y_test,"num2")


def process(path):
    data = pd.read_csv(path, dtype='a')
    label = np.array(data['emotion'])
    img_data = np.array(data['pixels'])
    N_sample = label.size
    print(N_sample)
    # print label.size
    Face_data = np.zeros((N_sample, a*b*c))
    Face_label = np.zeros((N_sample, 10), dtype=int)
    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ')
        Face_data[i] = x
        Face_label[i, int(label[i])] = 1
    return Face_data,Face_label
#train_num = 163
#test_num = 50
traindata,trainlabel=process("/home/xgd/lijiawei/num1.csv")
testdata,testlabel=process("/home/xgd/lijiawei/num2.csv")
################################
N1 = a*b  #  # of nodes belong to each window
N2 = c #  # of windows -------Feature mapping layer
N3 = 8000 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.9 #  shrink coefficient
C = 2**-30 # Regularization coefficient 

print('-------------------BLS_BASE---------------------------')
BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
print('-------------------BLS_ENHANCE------------------------')
BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)