import csv
import os
import sklearn
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda
from keras.layers import Convolution2D,MaxPooling2D,Cropping2D
from random import shuffle


# Importing images paths and angle values from csv file
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
      if (i ==1):
        samples.append(line)
      else:
        i = 1

#Spliting data to training and validaion sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Batch generation function
def generator(samples, batch_size=128):
  num_samples = len(samples)
  # correction angle for left and right cameras
  correction = 0.2 
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_samples:
        # Reading images in the batch
        center_image = cv2.imread('data/'+batch_sample[0])
        left_image = cv2.imread('data/'+batch_sample[1].strip())
        right_image = cv2.imread('data/'+batch_sample[2].strip())
        # Reading centre angle and computing left and right angles
        center_angle = float(batch_sample[3])
        left_angle = center_angle + correction
        right_angle = center_angle - correction
        # Adding Images
        images.append(center_image)
        images.append(left_image)
        images.append(right_image)
        # Augmenting data by flipping images
        images.append(cv2.flip(center_image,1)) 
        images.append(cv2.flip(left_image,1))
        images.append(cv2.flip(right_image,1))      
        # Adding angles
        angles.append(center_angle)
        angles.append(left_angle)
        angles.append(right_angle)
        # Adding augmented iamges angles
        angles.append(center_angle*-1.0)
        angles.append(left_angle*-1.0)
        angles.append(right_angle*-1.0)
        
    # passing the training and output values from the samples
    X_train = np.array(images)
    y_train = np.array(angles)
    yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

# NVIDEA Convolutional Neural Network in Keras
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample = (2,2) ,activation = "relu"))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compiling model
model.compile(loss = 'mse',optimizer= 'adam')
model.fit_generator(train_generator, samples_per_epoch=
            2*3*len(train_samples), validation_data=validation_generator,
            nb_val_samples=2*3*len(validation_samples), nb_epoch=5)

#saving model
model.save('model.h5')
