#! /usr/bin/python3
# encoding = utf-8

from random import random
from math import exp

class NeuralNetwork():
    def __init__(self, layers_sizes, learning_rate):
        """
        layer_sizes: List of number of neurons in each hidden layer.
        len(layer_sizes) indicate how many hidden layers are in the network.
        
        learning_rate: the learning rate.
        """
        self.n_layers = len(layers_sizes) + 2 # + 2 for input and output layers.
        self.layers_sizes = layers_sizes
        self.learning_rate = learning_rate

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __error_derivative(self, expected, given):
        return - (expected - given)

    def __layer_sigma(self,layer,i):
        if (layer == self.n_layers - 1):
            return self.__sigmoid_derivative(self.a[layer][i]) * self.__error_derivative(self.output[i],self.a[layer][i])
        else:
            return self.__sigmoid_derivative(self.a[layer][i]) * sum([self.w[layer][p][i] * self.__layer_sigma(layer + 1, p) for p in range(0,self.layers_sizes[layer + 1])])

    def __layer_threshold_deltas_calculus(self):
        self.threshold_deltas += [[self.__layer_sigma(layer,i) for i in range(self.layers_sizes[layer])] for layer in range(1,self.n_layers)]

    def __adjust_thresholds(self):
        self.u = [[None]*self.layers_sizes[0]]
        self.u += [[self.u[i][j] - self.threshold_deltas[i][j] for j in range(self.layers_sizes[i])] for i in range(1,self.n_layers)]

    def __thresholds_adjustment(self):
        self.threshold_deltas = [[None]*self.layers_sizes[0]] # Fill the first row of threshold with None (The input thresholds).
        self.__layer_threshold_deltas_calculus()
        self.__adjust_thresholds()

    def __layer_weight_deltas_calculus(self):
        self.weight_deltas = [[[self.a[layer - 1][j] * self.__layer_sigma(layer,i) for j in range(self.layers_sizes[layer - 1])] for i in range(self.layers_sizes[layer])] for layer in range(1,self.n_layers)]
    
    def __adjust_weights(self):
        self.w = [[[self.w[k][i][j] - self.learning_rate * self.weight_deltas[k][i][j] for j in range(self.layers_sizes[k])] for i in range(self.layers_sizes[k+1])] for k in range(self.n_layers - 1)]

    def __weights_adjustment(self):
        self.weight_deltas = []
        self.__layer_weight_deltas_calculus()
        self.__adjust_weights()

    def __think(self):
        self.a = list([self.input]) # Initialize the output array (a), with the input array (input) in first position.
        self.a += [[self.__sigmoid(self.u[k][i] + sum([self.w[k - 1][i][j] * self.a[k - 1][j] for j in range(self.layers_sizes[k - 1])])) for i in range(self.layers_sizes[k])] for k in range(1,self.n_layers)]

    def __generate_random_weigths(self):
        self.w = [[[random() for j in range(self.layers_sizes[k])] for i in range(self.layers_sizes[k + 1])] for k in range(self.n_layers - 1)]
        self.u = [None]*self.layers_sizes[0]
        self.u += [[random() for j in range(self.layers_sizes[i])] for i in range(1, self.n_layers)]

    def train(self, inputs, outputs, times):
        self.layers_sizes = [len(inputs[0])] + self.layers_sizes + [len(outputs[0])]
        self.__generate_random_weigths()
        train_cases = len(inputs) # Should be len(outputs).
        for x in range(times):
            for i in range(train_cases):
                self.input = inputs[i]
                self.output = outputs[i]
                self.__think()
                self.__weights_adjustment()
                self.__thresholds_adjustment()

    def predict(self,input,output):
        self.input = input
        self.__think()
        print("Expected: ", output)
        print("Given: ", self.a[self.n_layers-1])

if __name__ == "__main__":

    train_inputs = [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]
    test_inputs = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
    train_outputs = [[0],[1],[1],[1]]
    test_outputs = [[1],[1],[1],[1]]

    
    neuralNet = NeuralNetwork(
        layers_sizes = [4,4],
        learning_rate = 0.01
        )

    neuralNet.train(
        train_inputs,
        train_outputs,
        times = 1000
        )
    print("Ready")
    # neuralNetwork.predict([0,0,1],[1])
