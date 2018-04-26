#! /usr/bin/python3
# encoding = utf-8

from numpy import exp, array, random

class NeuralLayer():
    # def __init__(self,number_of_neurons,number_of_inputs_per_neuron):
    #     self.w = list()
    #     self.w = 2 * random.random((number_of_neurons,number_of_inputs_per_neuron)) - 1 # Comment this line on tests.


    #     self.n = int()
    #     self.n = number_of_neurons
        
    #     self.u = list()
    #     self.u = random.random(number_of_neurons) # Comment this line on tests.

        
        # print("w",self.w)
        # print("n",self.n)
        # print("u",self.u)

class NeuralNetwork():
    # def __init__(self,inputs,layers,outputs,learning_rate):
    #     self.w = list()
    #     self.w = [layer.w for layer in layers]

    #     self.inputs = inputs
    #     self.outputs = outputs

    #     self.u = list()
    #     self.u.append([None]*len(inputs[0]))
    #     self.u += [layer.u for layer in layers]

    #     self.n = list()
    #     self.n.append(len(inputs[0]))
    #     self.n += [layer.n for layer in layers]

    #     self.total_layers = len(layers) + 1
    #     self.learning_rate = learning_rate
        

    # def __sigmoid(self, x):
    #     return 1 / (1 + exp(-x))

    # def think(self):
    #     for k in range(1,self.total_layers):
    #         a = list()
    #         for i in range(0,self.n[k]):
    #             x = int()
    #             for j in range(0,self.n[k-1]):
    #                 x += self.w[k-1][i,j] * self.a[k-1][j]
    #             x += self.u[k][i]
    #             a.append(self.__sigmoid(x))
    #         self.a.append(a)

    def predict(self,input,output):
        self.input = input
        self.a = list()
        self.a.append(self.input)
        self.think()
        print(self.a[self.total_layers-1])

    # def __sigmoid_derivative(self,x):
    #     return x * (1 - x)

    # def __error_derivative(self,expected,given):
    #     return - (expected - given)

    # def __layer3_sigma(self,i):
    #     # print("a:",self.a)
    #     # print("total_layers:",self.total_layers)
    #     # print("output:",self.output)
    #     return self.__sigmoid_derivative(self.a[self.total_layers-1][i]) * self.__error_derivative(self.output[i],self.a[self.total_layers-1][i])

    # def __layer2_sigma(self,k):
    #     layer2_ways_sum = int()
    #     for i in range(0,self.n[self.total_layers-1]):
    #         layer2_ways_sum += self.w[self.total_layers-2][i,k] * self.__layer3_sigma(i)

    #     return self.__sigmoid_derivative(self.a[self.total_layers-2][k]) * layer2_ways_sum

    # def __layer1_sigma(self,k):
    #     layer1_ways_sum = int()
    #     for p in range(0,self.n[self.total_layers-2]):
    #         layer1_ways_sum += self.w[self.total_layers-3][p,k] * self.__layer2_sigma(p)

    #     return self.__sigmoid_derivative(self.a[self.total_layers-3][k]) * layer1_ways_sum

    # def layer1_weight_deltas_calculus(self):
    #     layer1_delta = list()
    #     for k in range(0,self.n[self.total_layers-3]):
    #         deltas = list()
    #         for j in range(0,self.n[self.total_layers-4]):
    #             delta = int()
    #             delta = self.a[self.total_layers-4][j] * self.__layer1_sigma(k)
    #             deltas.append(delta)
    #         layer1_delta.append(deltas)

    #     self.weight_deltas.append(array(layer1_delta))

    # def layer2_weight_deltas_calculus(self):
    #     layer2_delta = list()
    #     for k in range(0,self.n[self.total_layers-2]):
    #         deltas = list()
    #         for j in range(0,self.n[self.total_layers-3]):
    #             delta = int()
    #             delta = self.a[self.total_layers-3][j] * self.__layer2_sigma(k)
    #             deltas.append(delta)
    #         layer2_delta.append(deltas)

    #     self.weight_deltas.append(array(layer2_delta))

    # def layer3_weight_deltas_calculus(self):
    #     layer3_delta = list()
    #     for i in range(0,self.n[self.total_layers-1]):
    #         deltas = list()
    #         for j in range(0,self.n[self.total_layers-2]):
    #             delta = int()
    #             delta = self.a[self.total_layers-2][j] * self.__layer3_sigma(i)
    #             deltas.append(delta)
    #         layer3_delta.append(deltas)

    #     self.weight_deltas.append(array(layer3_delta))

    # def adjust_weights(self):
    #     # print("w:",self.w)
    #     # print("weight_deltas:",self.weight_deltas)
    #     for k in range(0,self.total_layers-1):
    #         for i in range(0,self.n[k+1]):
    #             for j in range(0,self.n[k]):
    #                 self.w[k][i,j] = self.w[k][i,j] - self.learning_rate * self.weight_deltas[k][i,j]
    
    # def weights_adjustment(self):
    #     self.weight_deltas = list()
    #     self.layer1_weight_deltas_calculus()
    #     self.layer2_weight_deltas_calculus()
    #     self.layer3_weight_deltas_calculus()
    #     self.weight_deltas = array(self.weight_deltas)
    #     self.adjust_weights()

    # def layer1_threshold_deltas_calculus(self):
    #     layer1_threshold = list()
    #     for i in range(0,self.n[self.total_layers-3]):
    #         delta = int()
    #         delta = self.__layer1_sigma(i)
    #         layer1_threshold.append(delta)
    #     self.threshold_deltas.append(layer1_threshold)

    # def layer2_threshold_deltas_calculus(self):
    #     layer2_threshold = list()
    #     for i in range(0,self.n[self.total_layers-2]):
    #         delta = int()
    #         delta = self.__layer2_sigma(i)
    #         layer2_threshold.append(delta)
    #     self.threshold_deltas.append(layer2_threshold)

    # def layer3_threshold_deltas_calculus(self):
    #     layer3_threshold = list()
    #     for i in range(0,self.n[self.total_layers-1]):
    #         delta = int()
    #         delta = self.__layer3_sigma(i)
    #         layer3_threshold.append(delta)
    #     self.threshold_deltas.append(layer3_threshold)
        
    # def adjust_thresholds(self):
    #     # print("u:",self.u)
    #     # print("threshold_deltas:",self.threshold_deltas)
    #     for i in range(1,self.total_layers):
    #         for j in range(0,self.n[i]):
    #             # print("position: (",i,j,")")
    #             self.u[i][j] = self.u[i][j] - self.learning_rate * self.threshold_deltas[i][j]

    # def thresholds_adjustment(self):
    #     self.threshold_deltas = list()
    #     self.layer0_threshold_deltas = [None]*self.n[0]
    #     self.threshold_deltas.append(self.layer0_threshold_deltas)
    #     self.layer1_threshold_deltas_calculus()
    #     self.layer2_threshold_deltas_calculus()
    #     self.layer3_threshold_deltas_calculus()
    #     self.adjust_thresholds()

    # def train(self,number_of_iterations):
    #     for x in range(0,number_of_iterations):
    #         for i in range(0,len(self.outputs)):
    #             self.a = list()
    #             self.a.append(self.inputs[i])
    #             self.output = self.outputs[i]

    #             self.think()
    #             self.weights_adjustment()
    #             self.thresholds_adjustment()

if __name__ == "__main__":

    train_inputs = [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]
    test_inputs = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
    layer1 = NeuralLayer(4,3)
    layer2 = NeuralLayer(4,4)
    layer3 = NeuralLayer(1,4)
    train_outputs = [[0],[1],[1],[1]]
    test_outputs = [[1],[1],[1],[1]]

    learning_rate = 0.01

    neuralNetwork = NeuralNetwork(train_inputs,[layer1,layer2,layer3],train_outputs,learning_rate)
    neuralNetwork.train(70000)
    print("Ready")
    neuralNetwork.predict([0,0,1],[1])
    
    # Uncomment this lines only during tests:
    # layer1 = NeuralLayer(3,2)
    # output_layer = NeuralLayer(1,3)
    # neuralNetwork = NeuralNetwork([2,3],[layer1,output_layer],None,None)
    # neuralNetwork.think()
    # print(neuralNetwork.w)
    # print(neuralNetwork.u)
    # print(neuralNetwork.n)
    # print(neuralNetwork.a)
