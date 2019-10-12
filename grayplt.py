# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:13:27 2019

@author: User
"""

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a functin do plot gray easily
def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot', interpolation='nearest')
    else:
        ax.imshow(img,cmap='hot', interpolation='nearest')
    
    plt.show()



raw_data2 = pd.read_excel('temp_data.xlsx',sheet_name = "temp_data")
sdarray2=raw_data2.drop(["ID","Time(min)","OvenTemp"],axis=1)

raw_label2=raw_data2.OvenTemp
sdarray_train2,sdarray_test2,label_train2,label_test2 = train_test_split(sdarray2,raw_label2,test_size = 0.01)

sdarray_train_np2=sdarray_train2.as_matrix()
#3.4 reshape into 2D array for each entry (shape=(entry,42,42))
twoDarray_train2 = sdarray_train_np2.reshape((sdarray_train_np2.shape[0], 32, 20))

print(twoDarray_train2[0].shape)


#Data print
for i in range(20):
    grayplt(twoDarray_train2[0])
