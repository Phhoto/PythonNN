import numpy as np

#3 samples 3 neurons
inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
           
biases = [2,3,0.5]

#reordered to inputs*weights so outputs looks like:
#first neuron outputs - - -
#second neuron
#third neuron etc
mult = np.dot(inputs, np.array(weights).T)
print(mult)
outputs = mult + biases
print(outputs)