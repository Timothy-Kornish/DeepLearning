# Mega Case Study - Make a Hybrid Deep Learning Model

# Part 1 - Identify the Frauds with the Sekf-Organizing Map

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pylab import bone, pcolor, colorbar, plot, show
from keras.models import Sequential
from keras.layers import Dense

# Importing util script for serializing model
import os
import sys
sys.path.append('../')
from util import ModelSerializer

# Import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
print(dataset.head())

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
scaler = MinMaxScaler(feature_range =(0, 1))
x = scaler.fit_transform(x)

# Training the SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

# visualize the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, j, in enumerate(x):
    w = som.winner(j)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(x)
# this value will change every random training run, must be manually observed then set,
# these coordinates are of red circle on green square
frauds = np.concatenate((mappings[(5, 2)], mappings[(4, 7)]), axis = 0)
frauds = scaler.inverse_transform(frauds)

# first number in tuple is number of frauds
print(frauds.shape)

# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# ANN

# Feature Scaling
scaler = StandardScaler()
customers = scaler.fit_transform(customers)

classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=15, units=6, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 10)

# Part 3 - Predicting the probabilites of frauds

# Predicting the results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred) , axis = 1)

# ranking probability of fraud
y_pred = y_pred[y_pred[:, 1].argsort()]
