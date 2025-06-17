import numpy as np


inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
           
biases = [2,3,0.5]

layer_outputs = []

# neurons = zip(weights, biases)
# #for each neuron (which has weights and a bais)
# for neuron_weights, neuron_bias in neurons:
#     #multiply each input by associated neuron weight
#     neuron_output = np.dot(inputs, neuron_weights)

#     #add bias, save
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

#numpy will dot product each entry of the 2d weights with the 1d input vector. then add biases
#whatever comes first in np.dot determines output shape
layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

#double brackets to define matrix with single row (defines dimensions - 2d)
#np.array([[1,2,3]])
#could also do:
#np.expand_dims(np.array([1,2,3]), axis=0)


a = [1,2,3]
b = [2,3,4]

a = np.array([a])
#transpose b
b = np.array([b]).T
#matrix multiplication
print(np.dot(a,b))