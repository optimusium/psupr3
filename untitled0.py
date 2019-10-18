# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:27:09 2019

@author: boonping
"""

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,RepeatVector
from tensorflow.keras.layers import Flatten,Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation,Dropout
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,SpatialDropout2D
from tensorflow.keras.layers import add,subtract,multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend 
import IPython

                            # Setting up the font manager, so that
                            # it can show japanese characters correctly
from matplotlib import font_manager as fm

raw_data = pd.read_excel('temp_data2.xlsx',sheet_name = "temp_data2")

#print (raw_data)
raw_temp=raw_data.drop(["ID","Time(min)","OvenTemp"],axis=1)
raw_output=raw_data.OvenTemp

temp=raw_temp.to_numpy()

#print( temp.shape )
#temp=temp.reshape(6,20,640)
temp=temp.reshape(200,20,16,20)
#print(temp[0])

temp2 = temp.transpose(0, 2, 3, 1)
print( np.max(temp2) )
print( np.min(temp2) )
print( np.max(temp2)-np.min(temp2) )
r=np.max(temp2)-np.min(temp2)
temp2= temp2 - np.min(temp2)
temp2=temp2/r

#print(temp2[0])
#print( temp2.shape )

for i in range(20):
    plt.plot(temp2[0][0][i])