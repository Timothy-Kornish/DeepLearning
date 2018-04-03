# import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class StackedAutoEncoder(nn.Module):

    def __init__(self, num_movies, nodes_first_hidden_layer =  20,
                nodes_second_hidden_layer = 10, nodes_third_hidden_layer = 20):
        """
        super():
            -- getting all variables and methods of parent class nn.Module

        full_connection_1:
            -- connection of nodes between input layer and first hidden layer

        full_connection_2:
            -- connection of nodes between first hidden layer and second hidden layer

        full_connection_3:
            -- connection of nodes between second hidden layer and third hidden layer

        full_connection_4:
            -- connection of nodes between third hidden layer and output layer

        activation:
            -- activation function (here using sigmoid)

        """
        super(StackedAutoEncoder, self).__init__()
        self.full_connection_1 = nn.Linear(num_movies, nodes_first_hidden_layer)
        self.full_connection_2 = nn.Linear(nodes_first_hidden_layer, nodes_second_hidden_layer)
        self.full_connection_3 = nn.Linear(nodes_second_hidden_layer, nodes_third_hidden_layer)
        self.full_connection_4 = nn.Linear(nodes_third_hidden_layer, num_movies)
        self.activation = nn.Sigmoid()

    def forward_propogation(self, input_vector):
        """
        input_vector:
            -- vector of input nodes (nodes in input layer)
        """
        input_vector = self.activation(self.full_connection_1(input_vector))
        input_vector = self.activation(self.full_connection_2(input_vector))
        input_vector = self.activation(self.full_connection_3(input_vector))
        input_vector = self.full_connection_4(input_vector)
        return input_vector
