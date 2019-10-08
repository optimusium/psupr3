# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:27:36 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:43:10 2019

@author: isstjh
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IPython

                            # Setting up the font manager, so that
                            # it can show japanese characters correctly
from matplotlib import font_manager as fm
fpath       = os.path.join(os.getcwd(), "ipam.ttf")
prop        = fm.FontProperties(fname=fpath)


                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'



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
        ax.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        ax.imshow(img,cmap='gray',vmin=0,vmax=1)
    circ=plt.Circle((21,21),radius=21.1,color='red',fill=False,linewidth=2.5)

    ax.add_patch(circ)
    
    plt.show()


# .............................................................................
raw_data = pd.read_excel('td8.xlsx',sheet_name = "td8")
sdarray=raw_data.drop(["saw","grind","dielectric"],axis=1)
raw_label=raw_data.saw+2*raw_data.grind+4*raw_data.dielectric

sdarray_train,sdarray_test,label_train,label_test = train_test_split(sdarray,raw_label,test_size = 0.2)

sdarray_train_np=sdarray_train.as_matrix()

twoDarray_train = sdarray_train_np.reshape((sdarray_train_np.shape[0], 42, 42))

sdarray_test_np=sdarray_test.as_matrix()

twoDarray_test = sdarray_test_np.reshape((sdarray_test_np.shape[0], 42, 42))

                            # Load the data
#trDat       = np.load('kmnist-train-imgs.npz')['arr_0']
trDat=twoDarray_train   
'''                         
print(trDat)
print(len(trDat))
print(trDat[0])
print(len(trDat[0]))
'''
trLbl=label_train
'''
print(trLbl)
print(len(trLbl))
'''
tsDat=twoDarray_test
tsLbl=label_test
#print(len(tsLbl))

#raise
                            # Convert the data into 'float32'
                            # Rescale the values from 0~255 to 0~1
#trDat       = trDat.astype('float32')/5 #/255
#tsDat       = tsDat.astype('float32')/5 #/255
trDat       = trDat.astype('float32')/500 #/255
tsDat       = tsDat.astype('float32')/500 #/255


                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = trDat.shape[1]
imgclms     = trDat.shape[2]


                            # reshape the data to be [samples][width][height][channel]
                            # This is required by Keras framework
trDat       = trDat.reshape(trDat.shape[0],
                            imgrows,
                            imgclms,
                            1)
tsDat       = tsDat.reshape(tsDat.shape[0],
                            imgrows,
                            imgclms,
                            1)


                            # Perform one hot encoding on the labels
                            # Retrieve the number of classes in this problem
trLbl       = to_categorical(trLbl)
tsLbl       = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

##--##
sdarray_temp=sdarray.as_matrix()
sdarray_temp=sdarray_temp.reshape((sdarray_temp.shape[0], 42, 42))
sdarray_temp       = sdarray_temp.astype('float32')/500 #/255

imgrows     = sdarray_temp.shape[1]
imgclms     = sdarray_temp.shape[2]


                            # reshape the data to be [samples][width][height][channel]
                            # This is required by Keras framework
sdarray_temp       = sdarray_temp.reshape(sdarray_temp.shape[0],
                            imgrows,
                            imgclms,
                            1)

#grayplt(tsDat[186])
#print (sdarray_temp)
#raise
def plotword(item,data=trDat,labels=trLbl):
    #clsname  = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']
    #clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']
    
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
        
    txt     = 'Class ' + str(lbl) + ': ' + clsname[lbl]     
    print(txt)
    grayplt(data[item],title=txt)

'''    
for i in range(0,1090,10):
    if raw_label[i]==4:
        print(i)
        plotword(i,sdarray_temp,raw_label)

raise
##--##

'''


# .............................................................................

                            # fix random seed for reproducibility
seed        = 29
np.random.seed(seed)


modelname   = 'wks5_5'
                            # define the deep learning model
def createModel():
    '''
    model = Sequential()       
    #model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation='relu'))       
    model.add(Conv2D(20, (4, 4), input_shape=(21, 21, 1), activation='relu'))       
    #model.add(MaxPooling2D(pool_size=(2, 2)))       
    model.add(MaxPooling2D(pool_size=(2, 2)))       
    model.add(Conv2D(40, (4, 4), activation='relu'))       
    #model.add(MaxPooling2D(pool_size=(2, 2)))       
    model.add(MaxPooling2D(pool_size=(2, 2)))       
  
    model.add(Dropout(0.2))       
    model.add(Flatten())       
    model.add(Dense(128, activation='relu'))       
    model.add(Dense(num_classes, activation='softmax'))            
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    '''
    inputShape=(42,42,1)
    inputs      = Input(shape=inputShape)
    #x           = Conv2D(128, (2, 2), padding="same",activation='relu')(inputs)
    x           = Conv2D(64, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(inputs)
    #x           = Lambda(lambda x: x  * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x           = Conv2D(60, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x)
    #x           = Conv2D(128, (2, 2), padding="same",activation='relu')(x)
    #x           = Lambda(lambda x: x  * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    encoded     = Conv2D(40, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x)
    #x           = Conv2D(32, (2, 2), padding="same",activation='relu')(x)
    #x           = Lambda(lambda x: x * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    #x           = Conv2D(8, (2, 2), padding="same",activation='relu')(x)
    #encoded     = Lambda(lambda x: x * 2, name="code")(x)
    
    x           = UpSampling2D(size=(2, 2))(encoded)
    x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice')(x) #
    x           = Conv2D(40, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #x           = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same")(x)
    #x           = Lambda(lambda x: x * 2)(x)
    x           = UpSampling2D(size=(2, 2))(x)
    x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice2')(x) #
    x           = Conv2D(60, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #x           = Lambda(lambda x: x * 2)(x)
    
    x           = UpSampling2D(size=(2, 2))(x)
    #x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice3')(x)
    outputs     = Conv2D(64, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #outputs     = Lambda(lambda x: x * 2)(x)
    
        
        
    y           = Flatten()(encoded)  
    y           = Dense(128, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Dense(128, activation='relu', kernel_initializer='he_normal')(encoded)
    y           = Dense(64, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Dense(32, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Flatten()(y)
    #y           = Dense(16, activation='relu', kernel_initializer='he_normal')(y)
    classifier  = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
      
    
    #outputs     = Dense(1, activation='softmax', kernel_initializer='he_normal')(x)  
    #outputs     = Dense(1, activation='relu', kernel_initializer='he_normal')(x)  
    #outputs     = Conv2D(1, (2, 2), padding="same",activation='relu')(x)
    
        
    #model       = Model(inputs=inputs,outputs=encoded)       
    model       = Model(inputs=inputs,outputs=outputs)       
    #model.compile(loss='binary_crossentropy', optimizer=adadelta, metrics=['accuracy'])   
    model.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['accuracy'])
   
    model2       = Model(inputs=inputs,outputs=classifier)       
    
    return model,model2

                          # Setup the models
model,model2       = createModel() # This is meant for training
modelGo,model2Go     = createModel() # This is used for final testing

model.summary()
model2.summary()
#print(len(model.get_weights()))
#raise
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#IPython.display.Image("model_plot.png")
#raise
# .............................................................................

def lrSchedule(epoch):
    lr  = 0.75e-3
    if epoch > 195:
        lr  *= 1e-4
    elif epoch > 180:
        lr  *= 1e-3
        
    elif epoch > 160:
        lr  *= 1e-2
        
    elif epoch > 140:
        lr  *= 1e-1
        
    elif epoch > 120:
        lr  *= 2e-1
    elif epoch > 60:
        lr  *= 0.5
        
        
    print('Learning rate: ', lr)
    
    return lr

def lrSchedule2(epoch):
    lr  = 0.4e-3
    if epoch > 59:
        lr  *= 1e-3
    elif epoch > 55:
        lr  *= 1e-2
        
    elif epoch > 52:
        lr  *= 2.5e-2
        
    elif epoch > 50:
        lr  *= 5e-2
        
    elif epoch > 48:
        lr  *= 0.1
    elif epoch > 45:
        lr  *= 0.5

        
        
    print('Learning rate: ', lr)
    
    return lr

LRScheduler     = LearningRateScheduler(lrSchedule)

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]




# .............................................................................


                            # Fit the model
                            # This is where the training starts

model_train=model.fit(trDat, 
            trDat, 
            validation_data=(tsDat, tsDat), 
            epochs=140, 
            batch_size=3,
            callbacks=callbacks_list)

loss=model_train.history['loss']
val_loss=model_train.history['val_loss']
epochs = range(140)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=20,
                             horizontal_flip=True,
                             vertical_flip=False)

model.fit_generator(datagen.flow(trDat, trLbl, batch_size=32),
                    validation_data=(tsDat, tsLbl),
                    epochs=200, 
                    verbose=1,
                    steps_per_epoch=len(trDat)/10,
                    callbacks=callbacks_list)

'''


# ......................................................................

                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='mean_squared_error', 
                optimizer=optimizers.RMSprop() ,
                metrics=['accuracy'])




# .......................................................................


for l1,l2 in zip(model2.layers[:3],model.layers[0:3]):
    l1.set_weights(l2.get_weights())
for layer in model2.layers[0:3]:
    layer.trainable = False
    
model2.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])

LRScheduler     = LearningRateScheduler(lrSchedule2)

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname+"_classifier" + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]


    
model2.fit(trDat, 
           trLbl, 
           validation_data=(tsDat, tsLbl), 
           epochs=60, 
           batch_size=10,
           callbacks=callbacks_list)


model2Go.load_weights(filepath)
model2Go.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])



                            # Make classification on the test dataset
predicts_img    = modelGo.predict(tsDat)
predicts    = model2Go.predict(tsDat)
#print(predicts[0])
#grayplt(tsDat[0])
#grayplt(predicts_img[0])


                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(tsLbl,axis=1)
#labelname   = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']
labelname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
                                            # the labels for the classfication report


testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)


print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)


# ...................................................................



def plotword(item,data=tsDat,data2=predicts_img,labels=tsLbl):
    #clsname  = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']
    #clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']
    
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
        
    txt     = 'Class ' + str(lbl) + ': ' + clsname[lbl]   
    print(txt)
    grayplt(data[item],title=txt)
    grayplt(data2[item],title=txt)
    
    
    
# ..................................................................
    
import pandas as pd

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.yticks([0.00,0.10,0.20,0.30])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.yticks([0.93,0.95,0.97,0.99])
plt.title('Accuracy',fontsize=12)
plt.show()

exampled=[]
for i in range(len(tsLbl)):
    #print(trLbl[i])
    for j in range(len(tsLbl[i])):
        if tsLbl[i][j]==1:
            break
    if j not in exampled:
        plotword(i)
        exampled.append(j)
    if len(exampled)==8: 
        break
        
'''
plotword(35)
plotword(235)
plotword(835)
plotword(635)
plotword(435)
'''
raw_data = pd.read_excel('td9.xlsx',sheet_name = "td9")
sdarray=raw_data

sdarray_train,sdarray_test,label_train,label_test = train_test_split(sdarray,sdarray,test_size = 0.9)

sdarray_test_np=sdarray_test.as_matrix()

twoDarray_test = sdarray_test_np.reshape((sdarray_test_np.shape[0], 42, 42))

tsDat=twoDarray_test

tsDat       = tsDat.astype('float32')/500 #/255

                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = tsDat.shape[1]
imgclms     = tsDat.shape[2]

tsDat       = tsDat.reshape(tsDat.shape[0],
                            imgrows,
                            imgclms,
                            1)

                            # Make classification on the test dataset
predicts_img    = modelGo.predict(tsDat)
predicts    = model2Go.predict(tsDat)


def plotword2(item,data=tsDat,data2=predicts_img,labels=tsLbl):
    #clsname  = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']
    #clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']
    
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
        
    txt     = 'On unlabel entry: Class ' + str(lbl) + ': ' + clsname[lbl]   
    print(txt)
    grayplt(data[item],title=txt)
    grayplt(data2[item],title=txt)

exampled=[]
for i in range(len(tsLbl)):
    #print(trLbl[i])
    for j in range(len(tsLbl[i])):
        if tsLbl[i][j]==1:
            break
    if j not in exampled:
        plotword2(i)
        exampled.append(j)
    if len(exampled)==8: 
        break
