import numpy as np
import pandas as pd
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Conv2D, Reshape
from keras.utils import Sequence
from keras.backend import epsilon
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import os


print(os.listdir(r'C:\Users\adars\OneDrive\Documents\deep learn\real_and_fake_face'))

real = r"C:\Users\adars\OneDrive\Documents\deep learn\real_and_fake_face\training_real"
fake = r"C:\Users\adars\OneDrive\Documents\deep learn\real_and_fake_face\training_fake"
real_path = os.listdir(real)
fake_path = os.listdir(fake)

def load_img(path):
    
    
    image = cv2.imread(path)
    image = cv2.resize(image,(224, 224))
    return image[...,::-1]

dataset_path = r"C:\Users\adars\OneDrive\Documents\deep learn\real_and_fake_face"

data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1./255,
                                   validation_split=0.2)



train = data_with_aug.flow_from_directory(dataset_path,
                                          
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="training")


val = data_with_aug.flow_from_directory(dataset_path,
                                        
                                        class_mode="binary",
                                        target_size=(96, 96),
                                        batch_size=32,
                                        subset="validation"
)

mnet = MobileNetV2(include_top = False, weights = "imagenet" ,input_shape=(96,96,3))

tf.keras.backend.clear_session()
model = Sequential([mnet,
                    
                    
                    GlobalAveragePooling2D(),
                    Dense(512, activation = "relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation = "relu"),
                    Dropout(0.1),
 # Dense(32, activation = "relu"),
# Dropout(0.3),
                    Dense(2, activation = "softmax")])

model.layers[0].trainable = False
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")
model.summary()

def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
       return 0.0001
    else:
      return 0.00001
lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

hist = model.fit_generator(train,
                           epochs=20,
                           callbacks=[lr_callbacks],
                           validation_data=val)

                                   
 
