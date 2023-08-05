import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint


# Constants
IMG_SIZE = 256
EPOCHS = 25
BATCH_SIZE = 20
SPLIT = 0.2


# Data Import
X = []
Y = []

data_dir = 'Chocolate Classification/'
classes = os.listdir(data_dir)


# Data Preparation
for i, name in enumerate(classes):
    images = glob(f'{data_dir}/{name}/*')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)
X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values

Data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.4)
])


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded_Y, test_size= SPLIT, random_state= 42, shuffle= True)


# Creating Model
model = keras.Sequential([
    Data_augmentation,
    
    layers.Conv2D(filters= 32, kernel_size= (5, 5), activation= 'relu', input_shape = (IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(pool_size= (2, 2), strides= 2, padding= 'same'),

    layers.Conv2D(filters= 64, kernel_size= (4, 4), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(pool_size= (2, 2), strides= 2, padding= 'same'),

    layers.Conv2D(filters= 128, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(pool_size= (2, 2), strides= 3, padding= 'same'),

    layers.Conv2D(filters= 256, kernel_size= (4, 4), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(pool_size= (2, 2), strides= 2, padding= 'same'),

    layers.Flatten(),
    layers.Dense(256, activation= 'relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation= 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])


# Callbacks
checkpoint = ModelCheckpoint('output/model.h5',
                             monitor= 'val_accuracy',
                             save_best_only= True,
                             save_weights_only= True,
                             verbose= 1)


# Model Training
history = model.fit(X_train, Y_train,
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    verbose= 1,
                    callbacks= checkpoint,
                    validation_data= (X_test, Y_test))