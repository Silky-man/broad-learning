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
from v2BroadLearningSystem import BLS,BLS_AddEnhanceNodes,BLS_AddFeatureEnhanceNodes,bls_train_input,bls_train_inputenhance

train_batch_size=64
datagen2 = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                            fill_mode='nearest')
datagen3 = ImageDataGenerator()
train_data=datagen2.flow_from_directory('/home/xgd/图片/carkeras/train',
                                  batch_size=train_batch_size,
                                  target_size=(224,224),
                                  shuffle=True)
val_data=datagen2.flow_from_directory('/home/xgd/图片/carkeras/val',
                                  batch_size=train_batch_size,
                                  target_size=(224,224),
                                  shuffle=True)
test_data=datagen3.flow_from_directory('/home/xgd/图片/carkeras/test',
                                       batch_size=200,
                                       target_size=(224,224),
                                       shuffle=False)
total_train=datagen3.flow_from_directory('/home/xgd/图片/carkeras/train',
                                  batch_size=1400,
                                  target_size=(224,224))
TIMESTAMP = "{0:%Y-%m-%dTime%H-%M-%S/}".format(datetime.now())
classes=10
img_input=layers.Input(shape=(224,224,3))

def identity_block(input_tensor, filters, stage, block):
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (3, 3),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',padding='same')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)


    x = layers.Conv2D(filters2, (3, 3),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',padding='same')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor,filters,stage,block,strides=(2, 2)):

    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (3, 3), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',padding='same')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, (3, 3),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',padding='same')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    shortcut = layers.Conv2D(filters2, (3, 3), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',padding='same')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet34(img_input):
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 64], stage=2, block='a',strides=(1, 1))
    x = identity_block(x, [64, 64], stage=2, block='b')
    x = identity_block(x, [64, 64], stage=2, block='c')

    x = conv_block(x,[128, 128], stage=3, block='a')
    x = identity_block(x, [128, 128], stage=3, block='b')
    x = identity_block(x, [128, 128], stage=3, block='c')
    x = identity_block(x, [128, 128], stage=3, block='d')

    x = conv_block(x, [256, 256], stage=4, block='a')
    x = identity_block(x, [256, 256], stage=4, block='b')
    x = identity_block(x, [256, 256], stage=4, block='c')
    x = identity_block(x, [256, 256], stage=4, block='d')
    x = identity_block(x, [256, 256], stage=4, block='e')
    x = identity_block(x, [256, 256], stage=4, block='f')

    x = conv_block(x, [512, 512], stage=5, block='a')
    x = identity_block(x, [512, 512], stage=5, block='b')
    x = identity_block(x, [512, 512], stage=5, block='c')

    x = layers.AveragePooling2D(pool_size=(7, 7),name='avg_pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(classes, activation='softmax',name='fc_4')(x)
    x = models.Model(img_input, x, name='resnet34')
    return x

model=ResNet34(img_input)
sgd=SGD(lr=0.001,momentum=0.9,decay=0.005)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit_generator(train_data,
                    steps_per_epoch=(train_data.samples//train_batch_size),
                    epochs=1,#120
                    validation_data=val_data,
                    validation_steps=(val_data.samples//train_batch_size))


encoder = models.Model(inputs=model.input, outputs=model.layers[-3].output)
train_images=encoder.predict(total_train[0][0])
print(train_images.shape)
_,a,b,c=train_images.shape
test_images=encoder.predict(test_data[0][0])
#################################################
def make(data11,label11,path):
    k=label11.shape[0]
    size=a
    data = np.zeros([k, size*size*c], dtype=np.uint8)
    label = np.zeros([k,10], dtype=int)
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
make(train_images,total_train[0][1],"num1")
make(test_images,test_data[0][1],"num2")


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
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.8 #  shrink coefficient
C = 2**-30 # Regularization coefficient 

print('-------------------BLS_BASE---------------------------')
BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
print('-------------------BLS_ENHANCE------------------------')
BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)










