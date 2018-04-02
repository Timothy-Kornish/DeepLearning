# Self Organizing map

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, colorbar, plot, show

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
frauds = np.concatenate((mappings[(8, 0)], mappings[(4, 9)]), axis = 0)
frauds = scaler.inverse_transform(frauds)
