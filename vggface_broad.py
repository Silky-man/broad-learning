# -*- coding:utf-8 -*-

from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import SGD
from datetime import datetime
import numpy as np
from keras.preprocessing import image
import os
import csv
import pandas as pd
from v1BroadLearningSystem import BLS

datagen = ImageDataGenerator()
train_data=datagen.flow_from_directory('/home/xgd/图片/lunwenyong/kerasyong/ck',
                                  batch_size=1191,
                                  target_size=(224,224),
                                  shuffle=False)
test_data=datagen.flow_from_directory('/home/xgd/图片/lunwenyong/kerasyong/jaffe',
                                       batch_size=213,
                                       target_size=(224,224),
                                       shuffle=False)

TIMESTAMP = "{0:%Y-%m-%dTime%H-%M-%S/}".format(datetime.now())
classes=7
img_input = layers.Input(shape=(224, 224, 3))

# Block 1
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
    img_input)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

# Block 2
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
    x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
    x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

# Block 3
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
    x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
    x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
    x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

# Block 4
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
    x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
    x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
    x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

# Block 5
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
    x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
    x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
    x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

model = models.Model(img_input, x, name='vggface_vgg16')  # load weights
model.load_weights('/home/xgd/lijiawei/rcmalli_vggface_tf_notop_vgg16.h5', by_name=True)

encoder = models.Model(inputs=model.input, outputs=model.layers[-1].output)
train_images=encoder.predict(train_data[0][0])
print(train_images.shape)
_,a,b,c=train_images.shape
test_images=encoder.predict(test_data[0][0])
#################################################
'''
def make(data11,label11,path):
    k=label11.shape[0]
    size=a
    data = np.zeros([k, size*size*c], dtype=np.uint8)
    label = np.zeros([k,7], dtype=int)
    for i in range(k):
        label[i] = label11[i]
        data[i][0:size*size*c] = np.ndarray.flatten(data11[i])

    with open("/home/xgd/lijiawei/"+path+".csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['emotion', 'pixels'])
        for i in range(len(label)):
            data_list = list(data[i])
            b = " ".join(str(x) for x in data_list)
            l = np.hstack([label[i], b])
            writer.writerow(l)
make(train_images,train_data[0][1],"num1")
make(test_images,test_data[0][1],"num2")
'''

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
N1 = 10#a*b  #  # of nodes belong to each window
N2 = 10#c #  # of windows -------Feature mapping layer
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.8 #  shrink coefficient
C = 2**-30 # Regularization coefficient 

print('-------------------BLS_BASE---------------------------')
BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
# print('-------------------BLS_ENHANCE------------------------')
# BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)
