    # Restricted Boltzmann Machines

# import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from RBM_class import RBM

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not liked), -1 (not rated)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Setting inputs to initialize the RBM
num_vis_nodes = len(training_set[0])
num_hid_nodes = 100
batch_size = 100

rbm = RBM(num_vis_nodes, num_hid_nodes)

#Training the RBM
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train_loss = 0
    counter = 0.0
    for id_user in range(0, nb_users - batch_size, batch_size):
        input_vector_k = training_set[id_user : id_user + batch_size]
        input_vector_ratings = training_set[id_user : id_user + batch_size]
        prob_h_equal_1_given_v,_ = rbm.sample_hidden_nodes(nodes = input_vector_ratings)
        for k in range(num_epochs):
            _,hidden_nodes_k = rbm.sample_hidden_nodes(nodes = input_vector_k)
            _,input_vector_k = rbm.sample_visible_nodes(nodes = hidden_nodes_k)
            input_vector_k[input_vector_ratings < 0] = input_vector_ratings[input_vector_ratings < 0]
        prob_h_equal_1_given_v_k,_ = rbm.sample_hidden_nodes(nodes = input_vector_k)
        rbm.train(input_vector = input_vector_ratings,
                  visible_nodes_k = input_vector_k,
                  prob_h_equal_1_given_v = prob_h_equal_1_given_v,
                  prob_h_equal_1_given_v_k = prob_h_equal_1_given_v_k)
        train_loss += torch.mean(torch.abs(input_vector_ratings[input_vector_ratings >= 0] - input_vector_k[input_vector_ratings >= 0]))
        counter += 1.
    print("\n Epoch: " + str(epoch) + " loss: " + str(train_loss/counter) + "\n")

# Testing the RBM
test_loss = 0
counter = 0.0
for id_user in range(nb_users):
    input_vector = training_set[id_user : id_user + 1]
    target_vector_ratings = test_set[id_user : id_user + 1]
    if(len(target_vector_ratings[target_vector_ratings >= 0]) > 0):
        _,hidden_nodes = rbm.sample_hidden_nodes(nodes = input_vector)
        _,input_vector = rbm.sample_visible_nodes(nodes = hidden_nodes)
        test_loss += torch.mean(torch.abs(target_vector_ratings[target_vector_ratings >= 0] - input_vector[target_vector_ratings >= 0]))
        counter += 1.
print("Test loss: " + str(test_loss/counter) + "\n")
