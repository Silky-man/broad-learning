#!/usr/bin/python
# coding:utf8

import cv2
import os
import numpy as np
import csv
from keras.models import Model
from keras import layers

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

model = Model(img_input, x, name='vggface_vgg16')  # load weights
model.load_weights(r'E:\dataset\newdirectory\vggface\rcmalli_vggface_tf_notop_vgg16.h5', by_name=True)

f = r"E:\dataset\vggboars\SFEW2.0\test"
fs = os.listdir(f)
np.random.shuffle(fs)
k=148
size=28
data = np.zeros([k, size*size*256], dtype=np.uint8)
label = np.zeros([k], dtype=int)
i = 0
for f1 in fs:
    tmp_path = os.path.join(f, f1)
    img = cv2.imread(tmp_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    layer_model = Model(inputs=model.input, outputs=model.layers[10].output)
    layer_output = layer_model.predict(img)
    img=layer_output[0]
        # 获得表情label
    img_label = f1[:2]
    print(img_label)
    if img_label == 'AN':
        label[i] = 0
    elif img_label == 'DI':
        label[i] = 1
    elif img_label == 'FE':
        label[i] = 2
    elif img_label == 'HA':
        label[i] = 3
    elif img_label == 'NE':
        label[i] = 4
    elif img_label == 'SA':
        label[i] = 5
    elif img_label == 'SU':
        label[i] = 6
    # elif img_label == '7':
    #     label[i] = 7
    else:
        print("get label error.......\n")
    data[i][0:size*size*256] = np.ndarray.flatten(img)
    i = i + 1
    # if i==213:
    #     break
print(i)

with open(r"E:\sfewtest.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels'])
    for i in range(len(label)):
        data_list = list(data[i])
        b = " ".join(str(x) for x in data_list)
        l = np.hstack([label[i], b])
        writer.writerow(l)
