import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras import datasets, layers, models
class Model():
    def __init__(self):
        cnn = models.Sequential()
        cnn.add(layers.Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape=(250,250,3)))
        cnn.add(layers.MaxPooling2D(2,2))
        cnn.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape=(250,250,3)))
        cnn.add(layers.MaxPooling2D(2,2))
        cnn.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape=(250,250,3)))
        cnn.add(layers.MaxPooling2D(2,2))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dense(128,activation = 'relu'))
        cnn.add(layers.Dense(5,activation = 'softmax'))
        checkpoint_path = "training_2/cp.ckpt"
        cnn.load_weights(checkpoint_path)
        self.mod = cnn
    def predict(self, file_path):
        img = Image.open(file_path)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        pred = self.mod.predict(img)
        try:
            index = np.where(pred[0] == 1)[0][0]
        except Exception as e:
            index = 0
        print(index)
        return index