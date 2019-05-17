#!/usr/bin/python
# coding:utf8

import cv2
import os
import numpy as np
import csv



f = "/home/xgd/lijiawei/jaffe/train"
fs = os.listdir(f)
np.random.shuffle(fs)
np.random.shuffle(fs)
np.random.shuffle(fs)
k=107
size=56
#data = np.zeros([k, size*size*3], dtype=np.uint8)
data = np.zeros([k, size*size], dtype=np.uint8)
label = np.zeros([k], dtype=int)
i = 0
for f1 in fs:
    tmp_path = os.path.join(f, f1)
    if not os.path.isdir(tmp_path):
        img = cv2.imread(tmp_path,0)
        img = cv2.resize(img, (size,size))
        img_label = f1[:2]
        print(img_label)
        if img_label == 'an':
            label[i] = 0
        elif img_label == 'di':
            label[i] = 1
        elif img_label == 'ha':
            label[i] = 2
        elif img_label == 'ne':
            label[i] = 3
        else:
            print("get label error.......\n")
        #data[i][0:size*size*3] = np.ndarray.flatten(img)
        # lbp=local_binary_pattern(img,8,2,'uniform')
        data[i][0:size * size] = np.ndarray.flatten(img)
        i = i + 1
    # if i==213:
    #    break
print(i)


with open("/home/xgd/lijiawei/jaffe.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels'])
    for i in range(len(label)):
        data_list = list(data[i])
        b = " ".join(str(x) for x in data_list)
        l = np.hstack([label[i], b])
        writer.writerow(l)