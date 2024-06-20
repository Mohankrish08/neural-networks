# importing libraries
import numpy as np

# single layer 
weights = [0.2, 0.4, 0.6, 0.8]
inputs = [1,2,3,4]
bias = 2

# dot product
output = np.dot(inputs, weights) + bias

print(output)

# multiple layer 
weights = [[0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.9, 0.12], [0.4, 0.8, 0.12, 0.16]]
biases = [2, 2.4, 4]

# dot product
layer_out = []
for weight, bias in zip(weights, biases):
    layer_out.append(np.dot(inputs, weight) + bias)

print(layer_out)