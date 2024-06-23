# importing libraries
import numpy as np

# Dense layer
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * (np.random.randn(n_inputs, n_neurons))
        self.bias = np.zeros((1, n_neurons))
    def forward(self, n_inputs):
        self.output = np.dot(n_inputs, self.weights) + self.bias

# creating activation
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# creating data
def creating_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype=np.uint8)
    for class_num in range(classes):
        ix = range(points*class_num, points*(class_num+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_num*4, (class_num+1)*4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_num
    return X, y

X,y = creating_data(100,4)

layer1 = LayerDense(2,5)
activation1 = ActivationReLU()

layer1.forward(X)

activation1.forward(layer1.output)
print(activation1.output)
