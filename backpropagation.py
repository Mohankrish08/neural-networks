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
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)   

# Activation function
class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0

# Activation Softmax
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# Loss   
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


epochs = 500
learning_rate = 0.01

X, y = creating_data(100, 4)

dense_1 = LayerDense(2,5)
activation_1 = ActivationReLU()

dense_2 = LayerDense(5,4)
activation_2 = ActivationSoftmax()

loss_function = LossCategoricalCrossentropy()

for epoch in range(epochs):

    dense_1.forward(X)
    activation_1.forward(dense_1.outputs)

    dense_2.forward(activation_1.outputs)
    activation_2.forward(dense_2.outputs)

    loss = loss_function.calculate(activation_2.output, y)

    # Accuracy
    predictions = np.argmax(activation_2.output, axis=1)
    accuracy = np.mean(predictions == y)

    print(f'Epoch: {epoch + 1}, Loss: {loss}, Accuracy: {accuracy * 100:.2f}%')

    activation_2.backward(activation_2.output, y)
    dense_2.backward(activation_2.dinputs)
    activation_1.backward(dense_2.dinputs)
    dense_1.backward(activation_1.dinputs)

    dense_1.weights -= learning_rate * dense_1.dweights
    dense_1.bias -= learning_rate * dense_1.dbiases
    dense_2.weights -= learning_rate * dense_2.dweights
    dense_2.bias -= learning_rate * dense_2.dbiases


    