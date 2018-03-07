#-------------------------------------------------------------------------------
#		                 Artificial Neural Network
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#                              Load in Libraries
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import keras # runs with tensorflow in background
from keras.models import Sequential # Initialize Neural Network
from keras.layers import Dense # for layers of ANN

#-------------------------------------------------------------------------------
#                               Load in Dataset
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Churn_Modelling.csv')
print('\n-------------------------------------------------------------------\n')
print(dataset.head())
print('\n-------------------------------------------------------------------\n')
x = dataset.iloc[:, 3:13].values # will produce same result as below
# x = dataset[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
             #'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
print(x)
print('\n-------------------------------------------------------------------\n')
# y = dataset.iloc[:, 13].values # will produce same result as below
y = dataset[['Exited']]
print(y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                     Encode Data (geography and gender)
#-------------------------------------------------------------------------------

labelEncoder_geography = LabelEncoder()
x[:, 1] = labelEncoder_geography.fit_transform(x[:, 1])

labelEncoder_gender = LabelEncoder()
x[:, 2] = labelEncoder_gender.fit_transform(x[:, 2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()
x = x[:, 1:] # get rid of dummy repetitions

#-------------------------------------------------------------------------------
#                           Train Test Split
#-------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#-------------------------------------------------------------------------------
#               Feature Scaling, puts all variables on equal footing
#-------------------------------------------------------------------------------

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#-------------------------------------------------------------------------------
#                    Initialize the Artificial Neural Network
#-------------------------------------------------------------------------------

classifier = Sequential()

#-------------------------------------------------------------------------------
#                      Add First (input) Layer to Neural Network
#-------------------------------------------------------------------------------

# Adding the input layer and the first hidden layer,
# 11 independent variable => input_dim = 11
# 6 nodes in hidden layer => output_dim = 6
# old format: classifier.add(Dense(output_dim = 6, init = 'uniform',
#                                  activation = 'relu', input_dim = 11))

# 11 independent variable => input_dim = 11
# 6 nodes in hidden layer => units= 6
# relu => recifier function
classifier.add(Dense(activation="relu", input_dim=11,
                     units=6, kernel_initializer="uniform"))

#-------------------------------------------------------------------------------
#                 Add Second and Third Layer to Neural Network
#-------------------------------------------------------------------------------

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#-------------------------------------------------------------------------------
#                 Add output Layer to Neural Network
#-------------------------------------------------------------------------------

# activation = 'softmax' is sigmoid function for dependent variable with more than 2 outcomes
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#-------------------------------------------------------------------------------
#                           Compiling the ANN
#-------------------------------------------------------------------------------

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#-------------------------------------------------------------------------------
#                 Fitting the training data to the ANN
#-------------------------------------------------------------------------------

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

#-------------------------------------------------------------------------------
#                 Making predictions with ANN
#-------------------------------------------------------------------------------

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print("Classification Report on Artifial Neural Network:\n",
       classification_report(y_test, y_pred))
print("\n-------------------------------------------------------------------\n")
print("Confusion Matrix on Artifial Neural Network:\n",
       confusion_matrix(y_test, y_pred))
print("\n-------------------------------------------------------------------\n")


#-------------------------------------------------------------------------------
#                 Making prediction on single new customer
#-------------------------------------------------------------------------------
"""
Info on customer
Geography: France, correspondes to dummy variables [0, 0] from encoding
CreditScore: 600
Gender: male (categorical, male = 1, female = 0)
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: yes (categorical, yes = 1, no = 0)
Is Active Member: yes (categorical, yes = 1, no = 0)
EstimatedSalary: 50000

info needs to be scaled
"""
customer = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
customer_scaled = scaler.fit_transform(customer)
new_prediction = classifier.predict(customer_scaled)
new_prediction = (new_prediction > 0.5)
print("Will Customer leave: ", new_prediction)
