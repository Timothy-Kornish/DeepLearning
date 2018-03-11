# Recurrent Neural Network
#-------------------------------------------------------------------------------
# Part 1 -  Data Preprocessing
#-------------------------------------------------------------------------------

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import os
import sys
sys.path.append('../')
from util import ModelSerializer



# Import the training_set
path = 'Google_Stock_Price_Train.csv'
stock_training = pd.read_csv(path)
training_set = stock_training.iloc[: , 1:2].values

# Feature Scaling
# Standardisation or Normalisation
# If output layer uses Sigmoid function, use Normalisation
scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
training_set_scaled = scaler.fit_transform(training_set)

# Create a Data structure with 60 timesteps and 1 output
x_train = []
y_train = []
days_of_memory = 60
for i in range(days_of_memory, 1258):
    x_train.append(training_set_scaled[i - days_of_memory: i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
print('x training set: \n', x_train)
print('y training set: \n', y_train)

# Reshaping array
batch_size = x_train.shape[0]
timesteps = x_train.shape[1]
input_dim = 1
x_train = np.reshape(x_train, (batch_size, timesteps, input_dim))

#-------------------------------------------------------------------------------
# Part 2 - Building the RNN
#-------------------------------------------------------------------------------

# Intiailize the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# Dropout reduces overfitting

units = 50 # number of units in layer, number of neurons per layer
return_sequences = True # trueb because this is a stacked LSTM RNN, multiple LSTM layers
input_shape = (timesteps, input_dim) # Here the shape will be 3D

regressor.add(LSTM(units = units, return_sequences = return_sequences, input_shape = input_shape))
regressor.add(Dropout(rate = 0.2))

# Adding the second LSTM layer and some Dropout regularisation
# No need to specify input_shape becuase units of previous layer is the next layers input_shape, which is already known
regressor.add(LSTM(units = units, return_sequences = return_sequences))
regressor.add(Dropout(rate = 0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units, return_sequences = return_sequences))
regressor.add(Dropout(rate = 0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = units, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Adding the ouput layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# optimizer for RNN on keras Docs: keras.optimizers.RMSprop(lr = 0.001, epsilon = 1e-08, decay = 0.0)
# optimizer Adam = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the RNN to the training set
batch_size = 32
regressor.fit(x = x_train, y = y_train, epochs = 100, batch_size = batch_size)

# Serialize model
ModelSerializer.serialize_model_json(regressor, 'RNN_4_layers_60_days', 'RNN_weights_4_layers_60_days')

#-------------------------------------------------------------------------------
# Part 3 - Making the predictions
#-------------------------------------------------------------------------------

# Loading in test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Get the predicted stock price of 2017
dataset_total = pd.concat((stock_training['Open'], dataset_test['Open']), axis = 0) # concat columns, axis = 0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - days_of_memory:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []

for i in range(days_of_memory, days_of_memory + 20):
    x_test.append(inputs[i - days_of_memory: i, 0])

x_test = np.array(x_test)

batch_size = x_test.shape[0]
timesteps = x_test.shape[1]
input_dim = 1
x_test = np.reshape(x_test, (batch_size, timesteps, input_dim))

predicted_stock_price = regressor.predict(x_test)

# invert the scaling to get the real world values
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#-------------------------------------------------------------------------------
# Part 4 - visualising the results
#-------------------------------------------------------------------------------

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Month of January')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
