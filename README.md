# PythonNN

train in batches which is faster and help with generalisation

ReLU - linear activation functions will produce linear outputs. ReLU(x) returns 0 if x < 0, or x otherwise (bounds and linear func can be edited but this can also be handled by weights and biases). Generally faster than other non-linear functions

softmax - e^output to make everything pos, highlight difference between numbers. Divide by sum of all e^outputs in sample to normalise

categorical cross-entropy - system outputs probs ie [0.7,0.2,0.1], one outputs is correct categorisation ie [1,0,0], loss is -log(0.7)
could get more complex if non-categorical, ie out = [0.7,0.2,0.1] exp = [0.6,0.1,0.3], loss = -(0.6*log(0.7) + 0.1*log(0.2) + 0.3*log(0.1))

the gradient ∇ - vector of all the partial derivatives of a function ie ∇f(x,y,z) = [f'wrt(x), f'wrt(y), f'wrt(z)]

stochastic gradient descent SGD - we have partial derivatives for literally everything that goes into f(...) = loss. Current values of any/all features 'x' (weight, bias) are plugged in in the backwards pass, solving the partial derivatives for the current values. This tells us the current gradient of loss wrt to each x? We want loss to be near 0. Simplest optimisation is to minus a fraction of each partial derivative from the related weight/bias so that the related weight/bias is contributing less to loss?
    parameter values - learning_rate*parameter_gradients