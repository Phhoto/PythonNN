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
        #remember inputs
        self.inputs = inputs
        #no need to transpose weights as we have set them up 'transposed'
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self,dvalues):
        #partial derivatives wrt each thing. Grow with batch size
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        #non-positive outputs are set to 0, ie neuron not firing
        #(what if we like negatives? weights will sort this out for us)
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self,inputs):
        #e^ everything
        #-max to make everything less than 0. Normalisation will reverse this anyway
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims=True))
        #axis=1 = sum column-wise (ie go down each entry in the column and sum the row?)
        #keepdims = [[1,2,3][4,5,6]] -> [[6],[15]] instead of [6, 15] 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        num_samples = len(y_pred)
        #clip data to prevent log(0)
        clipped_pred = np.clip(y_pred, 1e-7, 1 - (1e-7))

        #if 1d labels ie [2,0,1]
        if len(y_true.shape) == 1:
            #from each row grab confidence value for true val
            #for 2d array you can input 2 arrays of indexes
            correct_confidences = clipped_pred[range(num_samples), y_true]

        #if 2d labels ie [[0,0,1],[1,0,0],[0,1,0]]
        elif len(y_true.shape) == 2:
            #arr*arr does not do matrix multiplication? use np.dot? or maybe only in sum
            correct_confidences = np.sum(clipped_pred*y_true, axis=1)

        return -np.log(correct_confidences)
    
    def backward(self,dvalues,y_true):
        num_samples = len(dvalues)
        num_labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]

        #f(x) = -log(x) -> f'(x) = -1/x
        self.dinputs = -y_true / dvalues
        #average gradient
        self.dinputs = self.dinputs / num_samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

coords, y = spiral_data(samples=100, classes=3)

#input features are x and y coordinates off our spiral plot
dense1 = Layer_Dense(2,3)
dense2 = Layer_Dense(3,3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_calculator = Loss_CategoricalCrossEntropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
dense1.forward(coords)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
#activation2.forward(dense2.output)
#loss = loss_calculator.calculate(activation2.output, y)
loss = loss_activation.forward(dense2.output, y)

print(loss)

predictions = np.argmax(loss_activation.activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('acc: ', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
