# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:05:54 2022

@author: pmcen
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# download and store data as a directory 
import pathlib
# training set 
train_dataset_url = "https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz"
train_data_dir = tf.keras.utils.get_file('images', origin=train_dataset_url, untar=True)
#train_data_dir1 = pathlib.Path(train_data_dir)
# data wasn't reading in from the site correctly, below path link works 
train_data_dir = pathlib.Path("C:/Users/pmcen/OneDrive/Documents/college/nci/semester3/data mining and machine learning 2/project/fungi_train_val/images")

# testing set 
test_dataset_url = "https://labs.gbif.org/fgvcx/2018/fungi_test.tgz"
test_data_dir = tf.keras.utils.get_file('test', origin="file://C:/Users/pmcen/OneDrive/Documents/college/nci/semester3/data mining and machine learning 2/project/fungi_test/fungi_test/images")
#test_data_dir = tf.keras.utils.get_file('fungi_test', origin=test_dataset_url, untar=True)
test_data_dir = pathlib.Path(test_data_dir)
test_data_dir = pathlib.Path("file://C:/Users/pmcen/OneDrive/Documents/college/nci/semester3/data mining and machine learning 2/project/fungi_test/fungi_test/test")

image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(image_count)
# data cleaning, 
# most classes not loaded to directory in pathlib.Path


batch_size = 16
img_h = 512
img_w = 512 

train, val = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    subset = "both",
    seed = 21176671,
    validation_split = 0.2,
    image_size = (img_h, img_w),
    batch_size = batch_size)
#train = train.sample(frac=0.3, random_state=21176671)                   
class_names = train.class_names

"""
# init visualisations :
plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
"""


# performance improving config
AUTOTUNE = tf.data.AUTOTUNE
#train = train.cache().shuffle(10000).prefetch(buffer_size=AUTOTUNE)  # keeps images in memory after first load off disk in first Epoch and prevents any bottleneck with memory 
#val = val.cache().prefetch(buffer_size=AUTOTUNE) # overlaps preprocessing and execution of the model during training 
# running the above allows the model to run faster, but we don't have the ram locally to handle it so it crashes :( 

# Keras Model : 

num_classes = len(class_names)  # DOWNLOAD MORE RAM!!


model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_h, img_w, 3)),   # standardises the model
  layers.Conv2D(16, 3, padding='same', activation='relu'),   # first convolution layer 
  layers.MaxPooling2D(),                                    # first pooling layer  
  layers.Conv2D(32, 3, padding='same', activation='relu'),  # 2nd convolution layer 
  layers.MaxPooling2D(),                                    # 2nd pooling layer
  layers.Conv2D(64, 3, padding='same', activation='relu'),  # 3rd conolution layer 
  layers.MaxPooling2D(),                                    # 3rd pooling layer 
  layers.Conv2D(128, 3, padding='same', activation='relu'),  # 3rd conolution layer 
  layers.MaxPooling2D(),                                    # 3rd pooling layer 
  layers.Conv2D(256, 3, padding='same', activation='relu'),  # 3rd conolution layer 
  layers.MaxPooling2D(),                                    # 3rd pooling layer 
  layers.Conv2D(512, 3, padding='same', activation='relu'),  # 3rd conolution layer 
  layers.MaxPooling2D(),                                    # 3rd pooling layer 
  layers.Flatten(),                                      # flattin
  layers.Dense(1024, activation='relu'),     
  layers.Dense(num_classes)  # fully connected layers 
])

# compiling the model:    
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
# dense layer too big, training over epochs will always crash
model_0 = Sequential([
    layers.Rescaling(1./255, input_shape=(img_h, img_w,3)),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(), 
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model_0.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_0.summary()

#training model over 10 epochs 
epochs=3
history = model_0.fit(
  train,
  validation_data=val,
  epochs=epochs
)
# RESOURCE EXHAUSTED ERROR HERE, MAY NEED TO DROP BATCH SIZE/REDUCE IMAGE SIZE TO ALLOW SCRIPT TO RUN 

#training accuracy:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, 100*np.array(acc), label='Training Accuracy')
plt.plot(epochs_range, 100*np.array(val_acc), label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("accuracy (%)")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel("Epoch")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()









