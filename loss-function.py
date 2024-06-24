# importing libraries
import numpy as np

# data
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

# Neural networks
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * (np.random.randn(n_inputs, n_neurons))
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

# activation function
class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# softmax
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# loss
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_cliped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_cliped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_cliped*y_true, axis=1)
        negative_loss_likelyhood = -np.log(correct_confidences)
        return negative_loss_likelyhood
    
X, y = creating_data(100, 4)

dense_1 = LayerDense(2,5)
activation_1 = ActivationRelu()

dense_2 = LayerDense(5,4)
activation_2 = ActivationSoftmax()

dense_1.forward(X)
activation_1.forward(dense_1.output)

dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)

print(activation_2.output[:5])

loss_function = LossCategoricalCrossentropy()
loss = loss_function.calculate(activation_2.output, y)

print(loss)