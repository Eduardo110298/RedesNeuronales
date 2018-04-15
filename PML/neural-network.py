#! /usr/bin/python3
# encoding = utf-8

from numpy import exp, array, random

class NeuralLayer():
    def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
        self.w = list()
        self.w = 2 * random.random((number_of_neurons,number_of_inputs_per_neuron)) - 1 # Comment this line on tests.
        # self.w = array([[1]*number_of_inputs_per_neuron]*number_of_neurons) # Uncomment this line on tests (All weigths are 1).

        self.n = int()
        self.n = number_of_neurons
        
        self.u = list()
        self.u = random(number_of_neurons) # Comment this line on tests.
        # self.u = [1]*number_of_neurons # Uncomment this line on tests (All thresholds are 1).

class NeuralNetwork():
    def __init__(self,inputs,layers,outputs,learning_rate):
        self.w = list()
        self.w = [layer.w for layer in layers]

        self.inputs = inputs
        self.outputs = outputs

        self.u = list()
        self.u.append([None]*len(inputs[0]))
        self.u += [layer.u for layer in layers]

        self.n = list()
        self.n.append(len(input[0]))
        self.n += [layer.n for layer in layers]

        self.total_layers = len(layers) + 1
        self.learning_rate = learning_rate
        

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def think(self):
        for k in range(1,self.total_layers-1):
            a = list()
            for i in range(0,self.n[k]):
                x = int()
                for j in range(0,self.n[k-1]):
                    x += self.w[k-1][i,j] * self.a[k-1][j]
                x += self.u[k][i]
                a.append(self.__sigmoid(x))
            self.a.append(a)

    def __sigmoid_derivative(x):
        return x (1 - x)

    def __error_derivative(expected,given):
        return - (expected - given)

    def __layer3_sigma(self,i):
        return self.__sigmoid_derivative(self.a[self.total_layers-1][i]) * self.__error_derivative(self.output[i],self.a[self.total_layers-1][i])

    def __layer2_sigma(self,k):
        layer2_ways_sum = int()
        for i in range(0,self.n[self.total_layers-1]):
            layer2_ways_sum += self.w[self.total_layers-2][i,k] * self.__layer3_sigma(i)

        return self.__sigmoid_derivative(self.a[self.total_layers-2][k]) * layer2_ways_sum

    def __layer1_sigma(self,k):
        layer1_ways_sum = int()
        for p in range(0,self.n[self.total_layers-2]):
            layer1_ways_sum += self.w[self.total_layers-3][p,k] * self.__layer2_sigma(p)

        return self.__sigmoid_derivative(self.a[self.total_layers-3][k]) * layer1_ways_sum

    def layer1_weight_deltas_calculus(self):
        layer1_delta = list()
        for k in range(0,self.n[self.total_layers-3]):
            delta = int()
            for j in range(0,self.n[self.total_layers-4]):
                delta += self.a[self.total_layers-4][j] * self.__layer1_sigma(k)
            layer1_delta.append(delta)

        self.weight_deltas.append(layer1_delta)

    def layer2_weight_deltas_calculus(self):
        layer2_delta = list()
        for k in range(0,self.n[self.total_layers-2]):
            delta = int()
            for j in range(0,self.n[self.total_layers-3]):
                delta += self.a[self.total_layers-3][j] * self.__layer2_sigma(k)
            layer2_delta.append(delta)

        self.weight_deltas.append(layer2_delta)

    def layer3_weight_deltas_calculus(self):
        layer3_delta = list()
        for i in range(0,self.n[self.total_layers-1]):
            delta = int()
            for j in range(0,self.n[self.total_layers-2]):
                delta += self.a[self.total_layers-2][j] * self.__layer3_sigma(i)
            layer3_delta.append(delta)

        self.weight_deltas.append(layer3_delta)

    def adjust_weights(self):
        for k in range(0,self.total_layers-1):
            for i in range(0,self.n[k+1]):
                for j in range(0,self.n[k]):
                    self.w[k][i,j] = self.w[k][i,j] - self.learning_rate * self.weight_deltas[k][i,j]
    
    def weights_adjustment(self):
        self.weight_deltas = list()
        self.layer1_weight_deltas_calculus()
        self.layer2_weight_deltas_calculus()
        self.layer3_weight_deltas_calculus()
        self.weight_deltas = array(self.weight_deltas)
        self.adjust_weights()

    def train(self,number_of_iterations):
        for x in range(0,number_of_iterations):
            for i in range(0,len(self.outputs)):
                self.a = list()
                self.a.append(inputs[i])
                self.output = self.outputs[i]

                self.think()
                self.weights_adjustment()
                # self.threshold_adjustment()

if __name__ == "__main__":

    inputs = [[0,0,0]]
    layer1 = NeuralLayer(4,3)
    layer2 = NeuralLayer(4,4)
    layer3 = NeuralLayer(1,4)
    outputs = [0]
    learning_rate = 0.001

    neuralNetwork = NeuralNetwork(inputs,[layer1,layer2,layer3],outputs,learning_rate)
    neuralNetwork.train(60000)
    
    # Uncomment this lines only during tests:
    # layer1 = NeuralLayer(3,2)
    # output_layer = NeuralLayer(1,3)
    # neuralNetwork = NeuralNetwork([2,3],[layer1,output_layer],None,None)
    # neuralNetwork.think()
    # print(neuralNetwork.w)
    # print(neuralNetwork.u)
    # print(neuralNetwork.n)
    # print(neuralNetwork.a)
