import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    #initialise random weights and zeroed biases based on number of inputs and neurons
    def __init__(self, n_inputs, n_neurons):
        #randn produces numbers distributed normally w mean 0 var 1
        # inputs number of rows, neurons number of columns
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #no need to transpose weights as we have set them up 'transposed'
        self.output = np.dot(inputs, self.weights) + self.biases

coords, y = spiral_data(samples=100, classes=3)

#input features are x and y coordinates off our spiral plot
dense1 = Layer_Dense(2,3)
dense1.forward(coords)
print(dense1.output[:5])
