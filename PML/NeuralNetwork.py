#! /usr/bin/python3
# encoding = utf-8

from random import random
from math import exp

class NeuralNetwork():

    def __init__(self, layers_sizes, learning_rate):
        """
        layers_sizes: List of number of neurons in each hidden layer.
        len(layers_sizes) indicate how many hidden layers are in the network.
        
        learning_rate: the learning rate.
        """
        self.n_layers = len(layers_sizes) + 2 # + 2 for input and output layers.
        self.layers_sizes = layers_sizes
        self.learning_rate = learning_rate

    def test(self,k,i):
        x = self.u[k][i] + sum([self.w[k - 1][i][j] * self.a[k - 1][j] for j in range(self.layers_sizes[k - 1])])
        print(x)
        return x

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __error_derivative(self, expected, given):
        return - (expected - given)

    def __layer_sigma(self,layer,i):
        return self.__sigmoid_derivative(self.a[layer][i]) * self.__error_derivative(self.output[i],self.a[layer][i]) if (layer == self.n_layers - 1) else self.__sigmoid_derivative(self.a[layer][i]) * sum([self.w[layer][p][i] * self.__layer_sigma(layer + 1, p) for p in range(0,self.layers_sizes[layer + 1])])

    ########################################################################################################################################################################################################################################################################################################################
    # Thresholds weights (u) adjustment process.

    def __layer_threshold_deltas_calculus(self):
        self.threshold_deltas += [[self.__layer_sigma(layer,i) for i in range(self.layers_sizes[layer])] for layer in range(1,self.n_layers)]

    def __adjust_thresholds(self):
        self.u = [[None]*self.layers_sizes[0]] + [[self.u[i][j] - self.learning_rate * self.threshold_deltas[i][j] for j in range(self.layers_sizes[i])] for i in range(1,self.n_layers)]

    def __thresholds_adjustment(self):
        self.threshold_deltas = [[None]*self.layers_sizes[0]] # Fill the first row of threshold with None (The input thresholds).
        self.__layer_threshold_deltas_calculus()
        self.__adjust_thresholds()

    ########################################################################################################################################################################################################################################################################################################################
    # Outputs weights (a) adjustment process.

    def __layer_weight_deltas_calculus(self):
        self.weight_deltas = [[[self.a[layer - 1][j] * self.__layer_sigma(layer,i) for j in range(self.layers_sizes[layer - 1])] for i in range(self.layers_sizes[layer])] for layer in range(1,self.n_layers)]
    
    def __adjust_weights(self):
        self.w = [[[self.w[k][i][j] - self.learning_rate * self.weight_deltas[k][i][j] for j in range(self.layers_sizes[k])] for i in range(self.layers_sizes[k+1])] for k in range(self.n_layers - 1)]

    def __weights_adjustment(self):
        self.weight_deltas = []
        self.__layer_weight_deltas_calculus()
        self.__adjust_weights()

    ########################################################################################################################################################################################################################################################################################################################

    def __think(self):
        """
        Generate the output.
        """
        self.a = [self.input] # Initialize the output array (a), with the input array (input) in first position.
        for k in range(1,self.n_layers):
            self.a.append([self.__sigmoid(self.test(k,i)) for i in range(self.layers_sizes[k])])

    def __generate_random_weights(self):
        """
        Generate random weights for thresholds and outputs.
        """
        self.w = [[[random() for j in range(self.layers_sizes[k])] for i in range(self.layers_sizes[k + 1])] for k in range(self.n_layers - 1)]
        self.u = [[None]*self.layers_sizes[0]] + [[random() for j in range(self.layers_sizes[i])] for i in range(1, self.n_layers)]

    def train(self, inputs, outputs, times):
        """
        Train the network.
        inputs: is a list of study cases (inputs do not represent the outputs of the input layer, but this one: inputs[i] does represent that).
        outputs: is a list with the output for each study case present in inputs.
        times: represent how many times the network's weights should be adjusted using that study cases.

        i.e.:
        inputs = [
        [0,0,0], Case 1
        [0,1,0], Case 2
        [1,0,0], Case 3
        [1,1,0]  Case 4
        ] Everyone of this inputs can be used to know how many neurons are in the input layer.
        outputs = [
        [0], Output for Case 1
        [1], Output for Case 2
        [1], Output for Case 3
        [1]  Output for Case 4
        ] Everyone of this output can be used to know how many neurons are in the output layer.

        neuralNetwork = NeuralNetwork(
            layers_sizes = [4,3], Two hidden layers with 4 and 3 neurons respectively.
            learning_rate = 0.001
            )

        neuralNetwork.train(
            inputs,
            outputs,
            times = 1000
            )
        """
        if (self.n_layers != len(self.layers_sizes)):
            self.layers_sizes = [len(inputs[0])] + self.layers_sizes + [len(outputs[0])]
        self.__generate_random_weights()
        train_cases = len(inputs) # Could be len(outputs).
        for x in range(times):
            for i in range(train_cases):
                self.input = inputs[i]
                self.output = outputs[i]
                self.__think()
                self.__weights_adjustment()
                self.__thresholds_adjustment().

    def predict(self,input,output):
        """
        Make an output for a specific study case.
        input: list of outputs of input layer.
        input: list of outputs of output layer.
        """
        self.input = input
        if (self.n_layers != len(self.layers_sizes)):
            self.layers_sizes = [len(input)] + self.layers_sizes + [len(output)]
            self.__generate_random_weights()
        self.__think()
        print("Expected: ", output)
        print("Given: ", self.a[self.n_layers-1])


if __name__ == "__main__":

    train_inputs = [[20,20,20,0],[0,0,20,1],[0,0,1,1],[999,999,999,0],[0,1,1,1],[1,0,0,0],[1,0,1,0],[1,1,0,1]]
    train_outputs = [[0],[1],[1],[1],[0],[1],[0],[0]]

    neuralNet = NeuralNetwork(
        layers_sizes = [3,3],
        learning_rate = 0.1
        )

    neuralNet.train(
        train_inputs,
        train_outputs,
        times = 5000
        )
    print("Ready")
    neuralNet.predict([0,1,1,0],[0])
    neuralNet.predict([1,0,1,1],[0])
    neuralNet.predict([1,1,1,0],[0])
    neuralNet.predict([1,1,1,1],[0])