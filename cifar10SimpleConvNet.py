# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 01:15:14 2020

@author: Tobias
"""

#imports
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt

import tensorflow.keras as keras
to_categorical = keras.utils.to_categorical
# Load and prepare the MNIST dataset
cifar = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
# Convert to from integers to floating-point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0

# Look at one of the images in the dataset
plt.imshow(x_train[0])

# Check the shape of the image
x_train[0].shape
# Data pre-processing
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

# One-hot encode target column
y_train = to_categorical (y_train)
y_test = to_categorical (y_test)

y_train[0]
#build a sequential model by stacking layers
model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=5, strides=1, activation ='relu',input_shape=(32,32,3)), #img size: 29x29
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'), #img size: 17x17
        keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'), #img size: 15x15
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'), #img size: 7x7
        keras.layers.Conv2D(32, kernel_size=1, strides=1, activation='relu'), #img size: 7x7
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'), #img size: 6x6
        keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'), #img size: 4x4
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='valid'), #img size: 3x3
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
        ])

# Compile the model
optAdam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=optAdam, loss='categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate the model
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)
model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=100)
model.evaluate(x_test, y_test, verbose=2)

#Predict single image
#img = cv2.imread('./test1/1.jpg')
#model = create_model()
#model.load_weights('./weight.h5')
#model.predict(img)
