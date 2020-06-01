###
# Copyright (2019). All Rights belongs to VARUN
# Use the code by mentioning the Credits
# Credit: github.com/t-varun
# Developer:
#
#               T VARUN
#
###

# Import the required libraries
import numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import os

# Import th data
train_name = 'train_data_cleaned.npy'
test_name = 'test_data_cleaned.npy'
data_train = np.load(train_name)
data_test = np.load(test_name)

# Declare the required arrays
train_imgs = []
train_label = []
test_imgs = []
test_label = []

# Class names 
path_train_main = './FinalImages/Train/'
classes_train = [os.path.join(path_train_main, f).split('/')[-1] for f in os.listdir(path_train_main)]
path_test_main = './FinalImages/Test/'
classes_test = [os.path.join(path_test_main, f).split('/')[-1] for f in os.listdir(path_test_main)]
class_names = np.unique(classes_train + classes_test)
# print(class_names)

# Input to the arrays
for item, index in data_train:
    train_imgs.append(item)
    train_label.append(index)
for item, index in data_test:
    test_imgs.append(item)
    test_label.append(index)

# Train and Test data
train_images = train_imgs[-50:]
train_labels = train_label[-50:]

# test_images = test_imgs[:-50]
# test_labels = test_label[:-50]

train_images = np.asarray(train_images)
test_images = np.asarray(test_imgs)

# Image Processing
train_images = train_images / 255.0

test_images = test_images / 255.0

# Sequential Model
# Dense Layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(60, 80)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_images, train_labels, epochs=30)

# Print the Summary
model.summary()

# Accuracy of the Model
test_loss, test_acc = model.evaluate(test_images, test_label)

# Print Test accuracy
print('Test accuracy:', test_acc)

# Make Predictions
predictions = model.predict([test_imgs])[1] 
predicted_label = class_names[np.argmax(predictions)]

# Compare the predictions
print("Predictions : ",predicted_label)                      
print("Actual : ",class_names[test_label[1]])               

##print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model')
plt.ylabel('Result')
plt.xlabel('Epochs')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()
