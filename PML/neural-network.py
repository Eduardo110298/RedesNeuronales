#! /usr/bin/python3
# encoding = utf-8

from numpy import exp, array, random

class NeuralLayer():
    def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
        """ 
        w Denote the weights matrix random at first.
        n Denote the number of neurons in this layer.
        u Denote a list with the threshold input in each neuron.
        """
        self.w = list()
        self.w = 2 * random.random((number_of_neurons,number_of_inputs_per_neuron)) - 1 # Comment this line on tests.
        # self.w = array([[1]*number_of_inputs_per_neuron]*number_of_neurons) # Uncomment this line on tests (All weigths are 1).

        self.n = int()
        self.n = number_of_neurons
        
        self.u = list()
        self.u = random(number_of_neurons) # Comment this line on tests.
        # self.u = [1]*number_of_neurons # Uncomment this line on tests (All thresholds are 1).

class NeuralNetwork():
    def __init__(self,input,layers,output,learning_rate):
        self.w = list()
        self.w = [layer.w for layer in layers]

        self.a = list()
        self.a.append(input)

        self.u = list()
        self.u.append([None]*len(input))
        self.u += [layer.u for layer in layers]

        self.n = list()
        self.n.append(len(input))
        self.n += [layer.n for layer in layers]

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def think(self):
        """
        Generate the output.
        """
        for k in range(1,len(self.n)):
            a = list()
            for i in range(0,self.n[k]):
                x = int()
                for j in range(0,self.n[k-1]):
                    x += self.w[k-1][i,j] * self.a[k-1][j]
                x += self.u[k][i]
                a.append(self.__sigmoid(x))
            self.a.append(a)

if __name__ == "__main__":
    # Uncomment this only during tests:
    
    # layer1 = NeuralLayer(3,2)
    # output_layer = NeuralLayer(1,3)

    # neuralNetwork = NeuralNetwork([2,3],[layer1,output_layer],None,None)
    # neuralNetwork.think()

    # print(neuralNetwork.a)
    # print(neuralNetwork.w)
    # print(neuralNetwork.u)
    # print(neuralNetwork.n)
