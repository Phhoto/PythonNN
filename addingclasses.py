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


class Activation_ReLU:

    def forward(self, inputs):
        #non-positive outputs are set to 0, ie neuron not firing
        #(what if we like negatives? weights will sort this out for us)
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self,inputs):
        #e^ everything
        #-max to make everything less than 0. Normalisation will reverse this anyway
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims=True))
        #axis=1 = sum column-wise (ie go down each entry in the column and sum the row?)
        #keepdims = [[1,2,3][4,5,6]] -> [[6],[15]] instead of [6, 15] 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        num_samples = len(y_pred)
        #clip data to prevent log(0)
        clipped_pred = np.clip(y_pred, 1e-7, 1 - (1e-7))

        #if 1d labels ie [2,0,1]
        if len(y_true.shape) == 1:
            #from each row grab confidence value for true val
            correct_confidences = clipped_pred[range(num_samples), y_true]

        #if 2d labels ie [[0,0,1],[1,0,0],[0,1,0]]
        elif len(y_true.shape) == 2:
            #arr*arr does not do matrix multiplication? use np.dot
            correct_confidences = np.sum(clipped_pred*y_true, axis=1)

        return -np.log(correct_confidences)

coords, y = spiral_data(samples=100, classes=3)

#input features are x and y coordinates off our spiral plot
dense1 = Layer_Dense(2,3)
dense2 = Layer_Dense(3,3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_calculator = Loss_CategoricalCrossEntropy()

dense1.forward(coords)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_calculator.calculate(activation2.output, y)

print(loss)

predictions = np.argmax(activation2.output, axis=1)
print(predictions)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('acc: ', accuracy)
