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

#Section 0: Import Libraries
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


from matplotlib import font_manager as fm
fpath       = os.path.join(os.getcwd(), "ipam.ttf")
prop        = fm.FontProperties(fname=fpath)

#Section 1: Gray Plot Setting to print out wafer resistance map

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
    #show the wafer boundary. Radius is 21.
    circ=plt.Circle((21,21),radius=20.9,color='red',fill=False,linewidth=2.5)

    ax.add_patch(circ)
    
    plt.show()


# .............................................................................
#Section 2: Unlabelled data processing 
#2.1 Source: 1090 unlabelled resistance data. Each line represnts 42x42 locations with resistance measurement. In 1D 
raw_data = pd.read_excel('td9.xlsx',sheet_name = "td9")
sdarray=raw_data
#2.2 8:2 train/test data split. Ignore the label data as autoencoder training is unsupervised.
sdarray_train,sdarray_test,label_train,label_test = train_test_split(sdarray,sdarray,test_size = 0.2)

#2.3 convert array to numpy array. still in 1D for each entry
sdarray_train_np=sdarray_train.as_matrix()
sdarray_test_np=sdarray_test.as_matrix()

#2.4 reshape into 2D array for each entry (shape=(entry,42,42))
twoDarray_train = sdarray_train_np.reshape((sdarray_train_np.shape[0], 42, 42))
twoDarray_test = sdarray_test_np.reshape((sdarray_test_np.shape[0], 42, 42))


trDat=twoDarray_train   
tsDat=twoDarray_test

#2.5 Scaling. 500ohm is the maximum resistance value possible. Change to float type
trDat       = trDat.astype('float32')/500 #/255
tsDat       = tsDat.astype('float32')/500 #/255

#2.6 Reshape
# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows     = tsDat.shape[1]
imgclms     = tsDat.shape[2]

trDat       = trDat.reshape(trDat.shape[0],
                            imgrows,
                            imgclms,
                            1)
tsDat       = tsDat.reshape(tsDat.shape[0],
                            imgrows,
                            imgclms,
                            1)
#------------------------------------------------------------------------------    
#Section 3: labelled data processing (for classifier training. It links the features identified by the autoencoder to the classification of the technician)
#3.1 Source: 1090 labelled resistance data. It is at the same size of unlabelled data. The label of "saw","grind","material" fails are done by technician.
#            Each line represnts 42x42 locations with resistance measurement as well as 3 column of labels. In 1D 
raw_data2 = pd.read_excel('td8.xlsx',sheet_name = "td8")
#split into resistance input data and label output data
sdarray2=raw_data2.drop(["saw","grind","dielectric"],axis=1)
#Define as 8 class as permutation of "saw","grind" and "material"
raw_label2=raw_data2.saw+2*raw_data2.grind+4*raw_data2.dielectric

#3.2 8:2 train/test data split
sdarray_train2,sdarray_test2,label_train2,label_test2 = train_test_split(sdarray2,raw_label2,test_size = 0.2)

#3.3 convert array to numpy array. still in 1D for each entry
sdarray_train_np2=sdarray_train2.as_matrix()
#3.4 reshape into 2D array for each entry (shape=(entry,42,42))
twoDarray_train2 = sdarray_train_np2.reshape((sdarray_train_np2.shape[0], 42, 42))

sdarray_test_np2=sdarray_test2.as_matrix()

twoDarray_test2 = sdarray_test_np2.reshape((sdarray_test_np2.shape[0], 42, 42))

trDat2=twoDarray_train2   
'''                         
print(trDat)
print(len(trDat))
print(trDat[0])
print(len(trDat[0]))
'''
trLbl2=label_train2
'''
print(trLbl)
print(len(trLbl))
'''
tsDat2=twoDarray_test2
tsLbl2=label_test2
#print(len(tsLbl))

#3.5 Scaling. 500ohm is the maximum resistance value possible. Change to float type
                            # Convert the data into 'float32'
                            # Rescale the values from 0~255 to 0~1
#trDat       = trDat.astype('float32')/5 #/255
#tsDat       = tsDat.astype('float32')/5 #/255
trDat2       = trDat2.astype('float32')/500 #/255
tsDat2       = tsDat2.astype('float32')/500 #/255


                            # Retrieve the row size of each image
                            # Retrieve the column size of each image
imgrows     = trDat2.shape[1]
imgclms     = trDat2.shape[2]

#3.6 Reshape (1 channel 2D for each entry)
                            # reshape the data to be [samples][width][height][channel]
                            # This is required by Keras framework
trDat2       = trDat2.reshape(trDat2.shape[0],
                            imgrows,
                            imgclms,
                            1)
tsDat2       = tsDat2.reshape(tsDat2.shape[0],
                            imgrows,
                            imgclms,
                            1)

#3.7 Label must be change into categorical data. This is only applicable to classfier training.
# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
trLbl2       = to_categorical(trLbl2)
tsLbl2       = to_categorical(tsLbl2)
num_classes = tsLbl2.shape[1]

##--##
'''
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
'''

#Section 4: Allow wafer map plotting with class name
def plotword(item,data=trDat2,labels=trLbl2):
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

#Section 5: define the deep learning model
def createModel():
    #5.1 Set input shape which is 1 channel of 42x42 2D image
    inputShape=(42,42,1)
    inputs      = Input(shape=inputShape)
    
    #5.2 Auto encoder
    #This is to process the input so the neccessary feature that can be used for image reconstruction can be gathered.
    #5.2.1 First hidden layer. 64 neurons, downsize the image to (21,21). Conv2D is chosen over maxPooling as it allow the filter kernel to be trained.
    x           = Conv2D(64, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(inputs)
    #x           = Lambda(lambda x: x  * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    
    #5.2.2 Second hidden layer. 60 neurons, downsize the image to (11,11)
    x           = Conv2D(60, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x)
    #x           = Conv2D(128, (2, 2), padding="same",activation='relu')(x)
    #x           = Lambda(lambda x: x  * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    
    #5.2.2 Code Layer. Third hidden layer. 40 neurons, downsize the image to (6,6). This is the "encoded" layer. 
    #      "Code" is in this layer,  which will be used for decoder and classifier.
    #      Although further image downside is possible, it results in poorer re-constructed image. Hence, retain in (6,6).
    #      This is possibly due to the saw and grind defects take place at the edge. 
    #      Further downsize makes it harder to distinguish material fails near to edge but not at the edge and those real saw and grind defects.
    #      40 neurons are set for both clearer reconstructed image and better classification.
    #      Reducing number of neuron will cause more mis-classification and increase the loss in re-constructed image.
    encoded     = Conv2D(40, (2, 2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x)
    #x           = Conv2D(32, (2, 2), padding="same",activation='relu')(x)
    #x           = Lambda(lambda x: x * 2)(x)
    #x           = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    #x           = Conv2D(8, (2, 2), padding="same",activation='relu')(x)
    #encoded     = Lambda(lambda x: x * 2, name="code")(x)
    
    #5.3 Auto decoder
    #5.3.1 First Upsampling: Upsampling using (2,2) size so image is now (12,12)
    x           = UpSampling2D(size=(2, 2))(encoded)
    #5.3.2 First slicing: Due to upsampling, addtional 1 row and 1 column at the edge. remove it before any convolution. (11,11)
    x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice')(x) #
    #5.3.3 First convolution: Upsampling has increase the size. Hence, just strides=(1,1) operation.
    #      Not using maxPooling as convolution allow filter value to be trained.
    x           = Conv2D(40, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #x           = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="same")(x)
    #x           = Lambda(lambda x: x * 2)(x)
    
    #5.3.4 Second Upsampling: Upsampling using (2,2) size so image is now (22,22)    
    x           = UpSampling2D(size=(2, 2))(x)
    #5.3.5 Second slicing: Due to upsampling, addtional 1 row and 1 column at the edge. remove it before any convolution. (21,21)
    x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice2')(x) #
    #5.3.6 Second convolution: Upsampling has increase the size. Hence, just strides=(1,1) operation.
    x           = Conv2D(60, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #x           = Lambda(lambda x: x * 2)(x)

    #5.3.7 Third Upsampling: Upsampling using (2,2) size so image is now (42,42). Fits back the image size.       
    x           = UpSampling2D(size=(2, 2))(x)
    #x           = Lambda(lambda x: x[:,:-1,:-1,:], name='slice3')(x)
    #5.3.8 Third convolution: Upsampling has increase the size. Hence, just strides=(1,1) operation. 
    #                         This is the final reconstructed "decoded" image.
    outputs     = Conv2D(64, (2, 2), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    #outputs     = Lambda(lambda x: x * 2)(x)
    
    #5.4 Classifier    
    #This is to fit the features captured by the autoencoder to class defined by user
    #5.4.1 Flatten the code
    #The code layer consists of 40 neurons. Each with 6x6 2D image.    
    #Flatten to 1D array (40x6x6 neurons. This means 1440 type of features are possible. They should be trained on how human being differentiate the 8 classes)
    y           = Flatten()(encoded)  
    #5.4.2 Dense layers. forming network that results in 8 classes. Dense is used since it is now a 1D input.
    y           = Dense(128, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Dense(128, activation='relu', kernel_initializer='he_normal')(encoded)
    y           = Dense(64, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Dense(32, activation='relu', kernel_initializer='he_normal')(y)
    #y           = Flatten()(y)
    #y           = Dense(16, activation='relu', kernel_initializer='he_normal')(y)
    #5.4.3 Final layer of classfier. Using softmax activation so the most probable class can be computed. Total weight is always 1 at each neuron.
    classifier  = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
          
    #5.5 Forming autoencoder
    #Compiling using RMS since this is not categorical data at the output.
    model       = Model(inputs=inputs,outputs=outputs)       
    model.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['accuracy'])
    
    #5.6 Forming classfier. Will be compiled later using categorical crossentropy.
    model2       = Model(inputs=inputs,outputs=classifier)       
    
    return model,model2

#5.7 Create train and predict models. 
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

#Section 6: Learning rate 
#6.1 For autoencoder 
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

#6.2 For autoencoder classfier
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
    elif epoch > 30:
        lr  *= 0.8
    elif epoch > 25:
        lr  *= 0.9

        
        
    print('Learning rate: ', lr)
    
    return lr

#general setting for autoencoder training model
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


#Section 7: Training autodecoder. 140 epoch. 
#Learning rate is set higher as it requires more time to converge.
# Fit the model
# This is where the training starts
model_train=model.fit(trDat, 
            trDat, 
            validation_data=(tsDat, tsDat), 
            epochs=140, 
            batch_size=3,
            callbacks=callbacks_list)

#7.1 Plotting loss curve for autoencoder.
loss=model_train.history['loss']
val_loss=model_train.history['val_loss']
epochs = range(140)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# ......................................................................

#Section 8: Prepare prediction to reconstruct the image. ModelGo hence loads the weights
#Learning rate is set higher as it requires more time to converge.

                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='mean_squared_error', 
                optimizer=optimizers.RMSprop() ,
                metrics=['accuracy'])




# .......................................................................

#Section 9: Training autoencoder.
#9.1 Get weights for encoder while disallow weight at encoder to be trained again.
#note that [0:3] layers are actually the same  layers that we had trained durting the autoencoder training.
#subsequent dense layers should be trained instead.
for l1,l2 in zip(model2.layers[:3],model.layers[0:3]):
    l1.set_weights(l2.get_weights())
for layer in model2.layers[0:3]:
    layer.trainable = False

#9.2 Compiling model using categorical entropy. 
#output of classifer is categorical and not an image.    
model2.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])

#using slower learning rate than in autodecoder as 4 layers already been trained.
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


#9.3 Set the epoch to 60. As the encoding part is already been trained, it should converge faster to user defined classes.    
model2.fit(trDat2, 
           trLbl2, 
           validation_data=(tsDat2, tsLbl2), 
           epochs=60, 
           batch_size=1,
           callbacks=callbacks_list)


#Section 10: Prepare prediction model for the classfier.
model2Go.load_weights(filepath)
model2Go.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])



#Section 11: Prediction. Both reconstructed image and classification.
#11.1 Confusion matrix: The prediction is made on labelled data now to give the performance matrix.
predicts_img    = modelGo.predict(tsDat2)
predicts    = model2Go.predict(tsDat2)
#print(predicts[0])
#grayplt(tsDat[0])
#grayplt(predicts_img[0])


# Prepare the classification output
# for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(tsLbl2,axis=1)
#name for all the classes.
labelname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
                                            # the labels for the classfication report

#forming confusion matrix against the lavel given by the technician.
testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)

#giving accuracy score.
print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)


# ...................................................................



def plotword(item,data=tsDat2,data2=predicts_img,labels=tsLbl2):
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']
    
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
    #giving label text    
    txt     = 'Class ' + str(lbl) + ': ' + clsname[lbl]   
    print(txt)
    #print original image
    grayplt(data[item],title=txt)
    #print reconstructed image
    grayplt(data2[item],title=txt)
    
    
    
# ..................................................................
    
#Plotting training result of autoencoder
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

#Showing image of each class before and after going thru autoencoder.
exampled=[]
for i in range(len(tsLbl2)):
    #print(trLbl[i])
    for j in range(len(tsLbl2[i])):
        if tsLbl2[i][j]==1:
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

#Section 11.2 Prediction on unlabelled data
# Make classification on the test dataset
predicts_img    = modelGo.predict(tsDat)
predicts    = model2Go.predict(tsDat)

#Plotting the image and reconstrued image and showing the predicted class for unlabelled images.
def plotword2(item,data=tsDat,data2=predicts_img,labels=predicts):
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']
    #print(labels[item])
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
    #print(lbl)
        
    txt     = 'On unlabelled entry: Class ' + str(lbl) + ': ' + clsname[ lbl ]   
    print(txt)
    grayplt(data[item],title=txt)
    grayplt(data2[item],title=txt)

exampled=[]

#11.2.1 Showing image of each class before and after going thru autoencoder. It will also show the prediction of class.
#print(predicts[0])
for i in range(len(predicts)):
    #print predict[i]
    #print(trLbl[i])
    #for j in range(len(predicts[i])):
    #    if predicts[i][j]==1:
    #        break
    lbl=0
    if np.size(predicts.shape) == 2:
        lbl  = np.argmax(predicts[i])
    else:
        lbl  = predicts[i]
        
    if lbl not in exampled:
        plotword2(i)
        exampled.append(lbl)
    if len(exampled)==8: 
        break
