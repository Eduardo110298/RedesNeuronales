#! /usr/bin/python3
# encoding = utf-8

from numpy import exp, array, random, dot

class NeuralLayer():
    def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
    	""" 
		w denote the weights matrix random at first.
		n denote the number of neurons in this layer.
		u denote a list with the threshold input in each neuron.
    	"""

        self.w = list()
        self.w = random(number_of_neuron,number_of inputs_per_neuron)
        
        self.n = int()
        self.n = number_of_neurons
        
        self.u = list()
        self.u = random(number_of_neuron,number_of inputs_per_neuron)

class NeuralNetwork():
	def __init(self,input,layers,output,learning_rate):
        """

        """
        self.w = list()
        self.w = [layer.w for layer in range(layers)]

        self.a = list()
        self.a.append(input)

        self.u = list()
        self.u = [layer.u for layer in range(layers)]

        self.n = list()
        self.n = [layer.n for layer in range(layers)]

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def think(self):
        for k