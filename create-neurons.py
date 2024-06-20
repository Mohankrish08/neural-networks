# importing libraries
import time

# inputs
inputs = [1,2,3,4]

# weights
w1 = [0.2, 0.4, 0.6, 0.8]
w2 = [0.3, 0.6, 0.9, 0.12]
w3 = [0.4, 0.8, 0.12, 0.16]

# bias 
b1 = 2
b2 = 2.4
b3 = 4

# calculation
man_start = time.time()
output = [inputs[0]*w1[0] + inputs[1]*w1[1] + inputs[2]*w1[2] + inputs[3]*w1[3] + b1,
          inputs[0]*w2[0] + inputs[1]*w2[1] + inputs[2]*w2[2] + inputs[3]*w2[3] + b2,
          inputs[0]*w3[0] + inputs[1]*w3[1] + inputs[2]*w2[3] + inputs[3]*w3[3] + b3]

print(output)
man_end = time.time()


# cummulative
weights = [[0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.9, 0.12], [0.4, 0.8, 0.12, 0.16]]
biases = [2, 2.4, 4]

# iteration
cum_start = time.time()
layer_output = []
for tot_weights, bias in zip(weights, biases):
    neuron_out = 0
    for n_inp, weight in zip(inputs, tot_weights):
        neuron_out += n_inp * weight
    neuron_out += bias
    layer_output.append(neuron_out)

print(layer_output)
cum_end = time.time()

print(f"Manual_cal: {man_end - man_start}")
print(f"Cummulative: {cum_end - cum_start}")