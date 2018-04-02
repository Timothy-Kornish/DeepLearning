# Creating the architecture of the Neural Network
import torch

class RBM():
    def __init__(self, num_visible_nodes, num_hidden_nodes):
        """
        weights:
            -- The weights of each node assigned a random weight with normal distribution

        bias_prob_h_given_v:
            -- The bias of probability of hidden nodes given probability of visible nodes

        bias_prob_v_given_h:
            -- The bias of probability of visible nodes given probability of hidden nodes
        """
        self.weights = torch.randn(num_hidden_nodes, num_visible_nodes)
        self.bias_prob_h_given_v = torch.randn(1, num_hidden_nodes)
        self.bias_prob_v_given_h = torch.randn(1, num_visible_nodes)

    def sample_hidden_nodes(self, nodes):
        """
        nodes:
            -- Neurons in the chosen layer

        self.weights.t():
            -- t() is transpose function called on weights matrix
        """
        weights_times_neurons = torch.mm(nodes, self.weights.t())
        activation = weights_times_neurons + self.bias_prob_h_given_v.expand_as(weights_times_neurons)
        prob_h_given_v = torch.sigmoid(activation)
        return prob_h_given_v, torch.bernoulli(prob_h_given_v)

    def sample_visible_nodes(self, nodes):
        weights_times_neurons = torch.mm(nodes, self.weights)
        activation = weights_times_neurons + self.bias_prob_v_given_h.expand_as(weights_times_neurons)
        prob_v_given_h = torch.sigmoid(activation)
        return prob_v_given_h, torch.bernoulli(prob_v_given_h)

    def train(self, input_vector, visible_nodes_k, prob_h_equal_1_given_v, prob_h_equal_1_given_v_k):
        """
        input_vector:
            -- vector containing ratings on all movies by one user

        visible_nodes_k:
            -- vector of visible nodes after k-samplings

        prob_h_equal_1_given_v:
            -- vector of probabilites the hidden nodes equal 1, given the input_vector

        prob_h_equal_1_given_v_k
            -- vector of probabilites of hidden nodes equal 1, given visible_nodes_k after k-sampling
        """
        self.weights += torch.mm(input_vector.t(), prob_h_equal_1_given_v) - torch.mm(visible_nodes_k.t(), prob_h_equal_1_given_v_k)
        self.bias_prob_v_given_h += torch.sum((input_vector - visible_nodes_k), 0) # This will keep the format dimesions
        self.bias_prob_h_given_v += torch.sum((prob_h_equal_1_given_v - prob_h_equal_1_given_v_k), 0)
