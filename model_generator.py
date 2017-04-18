# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:29:45 2017

@author: Aneeq Mahmood
@email: aneeq.sdc@gmail.com
"""

# Import general use modules
import numpy as np
import matplotlib.pyplot as plt


# Neural networks related modules
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Convolution2D,ZeroPadding2D,Cropping2D,Lambda,Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger

def pre_process(image):
    """
    A pre processing function to create a lambda layer inside keras 
    to perform image resizing and normalization 
    
    Argument: 
    image: an image file

    """
    
    import tensorflow as tf
    resized = tf.image.resize_images(image, (66,200))
    normalized = resized/255.0 - 1.0
    return normalized
 

# Making CNN


import images_aug as iaug

csv_content = iaug.read_csvfile()
print(np.shape(csv_content))




valid_size= int((0.2* len(csv_content)))
valid_data = []



for m  in range(valid_size):
    ind = np.random.randint(0,valid_size)
    valid_data.append(csv_content[ind])
    del csv_content[ind]
    
   

print(np.shape(csv_content))
print(np.shape(valid_data))

#import sys
#sys.exit()



#
#
#X_train= np.load("Xdata.npy")
#y_train= np.load("ydata.npy")
#X_train = X_train /255.0 - 1.0  

# Making CNN

#im_shape = np.shape(X_train[0])


import os
if os.path.exists("model_X.h5"):
    print("Using old weights ")
    model = load_model("model_X.h5")

else:
    
    print("Training a  new model")

    model = Sequential()
    
    # Add a layer to crop top and bottom pixels 
    model.add(Cropping2D(cropping=((74,20), (0,0)), input_shape=(160,320,3)))
    
   # Add a lambda layer to resize image acceptable for Nvidia model and normalize
    model.add(Lambda(pre_process))
        
    # Add zero padding around images
    model.add(ZeroPadding2D((3,3), input_shape=(66,200,3)))
    
    # perform 2D convolution
    model.add(Convolution2D(24, (5,5), strides=(2,2),padding = 'valid', activation='relu')  )
    
     # Add zero padding around images
    model.add( ZeroPadding2D((3,3)) )
    # perform 2D convolution
    model.add(Convolution2D(36, (5,5), strides=(2,2),padding = 'valid', activation='relu') )
    
    # Add zero padding around images
    model.add(ZeroPadding2D((3,3)))
    # perform 2D convolution
    model.add(Convolution2D(48, (5,5), strides=(2,2) ,padding = 'valid', activation='relu' ) )
    
    
    # perform 2D convolution with reduced filter size
    model.add(Convolution2D(64, (3,3), strides=(1,1), padding = 'valid', activation='relu'  ) )
    
    # perform 2D convolution with reduced filter size
    model.add(Convolution2D(64, (3,3), strides=(1,1), padding = 'valid', activation='relu'  ) )
    
    
    model.add(Flatten())
    # Fully connected and dropout layer
    model.add(Dense(1164))      
    model.add(Dropout(0.5))             
    
    # Fully connected and dropout layer
    model.add(Dense(100))
    model.add(Dropout(0.5))
    
    # Fully connected and dropout layer
    model.add(Dense(50))
    model.add(Dropout(0.5))
    
    # Fully connected output layera
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile( loss = 'mse',optimizer = 'adam'  )


#define batch size
batch_size = 32

# generators for training and validation data
train_gen = iaug.get_data_generator(csv_content, batch_size)
validation_gen = iaug.get_data_generator(valid_data, batch_size)


model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_gen, steps_per_epoch=batch_size*5, validation_data=validation_gen, validation_steps=batch_size*1, epochs=3,verbose = 1)

#model.fit(X_train,y_train,validation_split = 0.2, shuffle= True,  epochs = 3)

total_epochs = 5

model.fit_generator(train_gen,
                    steps_per_epoch=len(csv_content),
                    epochs=total_epochs,
                    verbose=1,
                    validation_data=validation_gen,
                    validation_steps=len(valid_data))

#
#EarlyStopping(monitor='val_loss',
#                                             min_delta=2,
#                                             patience=2,
#                                             verbose=2,
#                                             mode='min'),







model.save('model_X.h5')

     
     