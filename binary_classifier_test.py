from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


df = pd.read_csv("output_files/training_data/5_v6_big.feat", header=None)
dataset = df.values
X = dataset[:, 0:-1]
y = dataset[:, -1]
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.1)

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_dim=301),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epoch = 10
model.fit(X_train, y_train, epochs=epoch)
_, accuracy = model.evaluate(X_test, y_test)
model.save("models/binary_classifiers/{}_model.h5".format(epoch))

print('Accuracy: %.2f' % (accuracy * 100))
