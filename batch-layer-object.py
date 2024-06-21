# importing libraries
import numpy as np
np.random.seed(42)

# inputs
inputs = [[1,2,3,4],
          [2.5, 1.2, 3.6, 4.1],
          [1.3, 2.4, 2.8, 4.2]]
weights = [[0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.9, 0.12], [0.4, 0.8, 0.12, 0.16]]
bias = [3, 5, 0.3]

weights2 = [[0.1, -0.3, 0.5],
            [1.2, -2.3, 1.4],
            [0.3, 1.7, 5.2]]
bias2 = [0.5, 0.2, 0.1]

# result
layer1 = np.dot(inputs, np.array(weights).T) + bias
layer2 = np.dot(layer1, np.array(weights2).T) + bias2
print(layer1)
print("*" * 20)
print(layer2)


# objects

X = [[0.2, 0.4, -0.6, -0.8], 
     [0.4, -0.8, 0.12, -0.16],
     [0.12, 0.32, -0.4, -1.3]]


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

layer_1 = LayerDense(4,5)
layer_2 = LayerDense(5,1)

layer_1.forward(X)
print(layer_1.output)
print("*" * 40)
layer_2.forward(layer_1.output)
print(layer_2.output)