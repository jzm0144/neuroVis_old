# -*- coding: utf-8 -*-
"""
Created on Sunday October 1st 19:31:41 2019

@author: Janzaib Masood
"""



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import *
import ipdb as ipdb





def vecFix(a):
    vecLength = a.shape[0]
    sqrLength = sqrt(a.shape[0])

    picLength = ceil(sqrLength)

    if sqrLength**2 == picLength**2:
        out = a#.reshape((sqrt(picLength), sqrt(picLength))
    else:
        b = np.zeros(picLength**2)
        b[:vecLength] = a[:]
        out = b
    return out

def vec2Square(a):
    o = vecFix(a)
    return o.reshape((int(sqrt(o.shape[0])), int(sqrt(o.shape[0]))))



def shapeData(path):
    Q = pd.read_excel(path)
    Subjects   = Q.iloc[1:,2:].values
    numSubjects = Subjects.shape[1]
    numPaths   = Subjects.shape[0]

    picDim = vec2Square(Subjects[:,0]).shape[0]

    X      = np.zeros((numSubjects, picDim, picDim))
    Y      = Q.iloc[0,2:].values 
    for subjId in range(numSubjects):
        X[subjId, :, :] = vec2Square(Subjects[:,subjId])
    return X, Y

trainPath = "/Users/jzm0144/Janzaib_Playground/project_neuroVis/train_data.xlsx"
testPath = "/Users/jzm0144/Janzaib_Playground/project_neuroVis/test_data.xlsx"
xTrain, yTrain = shapeData(trainPath)
xTest, yTest   = shapeData(testPath)

yTrain[yTrain == 'Controls'] = 0
yTrain[yTrain == 'EMCI'] = 1
yTrain[yTrain == 'LMCI'] = 2
yTrain[yTrain == 'AD'] = 3

yTest[yTest   == 'Controls'] = 0
yTest[yTest   == 'EMCI'] = 1
yTest[yTest   == 'LMCI'] = 2
yTest[yTest   == 'AD'] = 3


# Brain all Data in range 0.0 and 1.0
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = (xTrain + 1)/2
xTest  = (xTest  + 1)/2



batch_size = 64
num_classes = 4
epochs = 64

# input image dimensions
img_rows, img_cols = xTrain.shape[1:]
print("Image Size = ("+str(img_rows)+", "+str(img_cols),")")


if K.image_data_format() == 'channels_first':
    xTrain = xTrain.reshape(xTrain.shape[0], 1, img_rows, img_cols)
    xTest = xTest.reshape(xTest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    xTrain = xTrain.reshape(xTrain.shape[0], img_rows, img_cols, 1)
    xTest = xTest.reshape(xTest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# convert class vectors to binary class matrices
yTrain = keras.utils.to_categorical(yTrain, num_classes)
yTest = keras.utils.to_categorical(yTest, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(xTrain, yTrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xTest, yTest))
score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])