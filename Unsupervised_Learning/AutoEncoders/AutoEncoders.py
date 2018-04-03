# Auto Encoders

# import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from AutoEncoder_class import StackedAutoEncoder

# Import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = "::", header = None,
                     engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = "::", header = None,
                     engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = "::", header = None,
                     engine = 'python', encoding = 'latin-1')

# preparing the training set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

# preparing the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

print(movies.head())
print(users.head())
print(ratings.head())
print("\n-------------------------------------------------------------------\n")

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# initializing Stacked AutoEncoder
SAE = StackedAutoEncoder(nb_movies)
criterion_loss_function = nn.MSELoss() # Mean Squared Error loss
optimizer = optim.RMSprop(SAE.parameters(), lr = 0.01, weight_decay = 0.5) # Root Mean Sqared, lr = learning rate,

# Training the Stacked Auto Encoder
num_epochs = 200

for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    RMSCounter = 0.0
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = SAE.forward_propogation(target)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion_loss_function(output, target)
            mean_corrector =  nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            RMSCounter += 1.
            optimizer.step()
    print("Epoch: " + str(epoch) + " Loss: " + str(train_loss/RMSCounter))

# Testing the Stacked Auto Encoder
test_loss = 0.0
RMSCounter = 0.0
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = SAE.forward_propogation(target)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion_loss_function(output, target)
        mean_corrector =  nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        RMSCounter += 1.
print("Loss: " + str(test_loss/RMSCounter))
