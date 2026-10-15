# ## Final Architecture

# from pyparsing import Forward


# Our architecture is designed to process multiple images simultaneously during both inference and training.

# ### 1. Input and forward propagation

# - Each image is represented as a vertical column of pixel values.
# - We can place multiple images side by side as columns in one input matrix.
# - Every layer processes all these columns using matrix multiplication, 
#     so each neuron produces an activation for every image.
# - Therefore, every layer's output is also a matrix, with each column corresponding to one image.

# ### 2. Output

# - The final layer produces one output column for each input image.
# - Each output column contains the predicted probabilities for all classes.
# - We can find the predicted class for each image and represent it as a one-hot column if required.

# ### 3. Batch training and backpropagation

# - We provide multiple images and their corresponding one-hot labels as input.
# - The network performs forward propagation on the entire batch.
# - We calculate the output error separately for every image, producing an error matrix where each column corresponds to one image.
# - We propagate these errors backward through the network, 
#     calculating the contribution of every image to the gradients of the weights and biases.
# - We then average these gradient contributions across all images in the batch.

# Important distinction: We do not simply average the output error columns first and then perform backpropagation. We perform backpropagation using the individual error columns and average their contributions to the parameter gradients.

# ### 4. Updating the network

# - Once we have calculated the average np.gradient for every weight and bias, 
#     we update all the parameters using those gradients and the learning rate.
# - This is one training update for the entire batch, regardless of how many images it contains.

# ### 5. Repeating training

# - We repeat this process for each batch until the entire training dataset has been processed. This completes one epoch.
# - We repeat for multiple epochs to train the network.
# - After training, we test it on separate images to evaluate its performance.

# In summary: Every image occupies one column throughout the network. 
#             Forward propagation and backpropagation operate on entire batches using matrices, while the parameter gradients are averaged across the images before a single update. The same architecture also supports batch inference, producing predictions for multiple images in one forward pass.

# notation:
# a0 matrix: (nn0, m) = (784, m)
# w1 matrix: (nn1, nn0) = (10, 784)
# b1 matrix: (nn1, 1) = (10, 1)
# z1 matrix = w1 * a0 + b1:  (nn1, m) = (10, m)

# a1 matrix = ReLU(z1): (nn1, m) = (10, m)
# w2 matrix: (nn2, nn1) = (10, 10)
# b2 matrix: (nn2, 1) = (10, 1)
# z2 matrix = w2 * a1 + b2:  (nn2, m) = (10, m)

# a2 matrix = softmax(z2): (nn2, m) = (10, m)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#functions:

# we initially need random weights and biases
def initialize_parameters(nn0, nn1, nn2):
    w1 = np.random.rand(nn1, nn0) - 0.5
    b1 = np.random.rand(nn1, 1) - 0.5
    w2 = np.random.rand(nn2, nn1) - 0.5
    b2 = np.random.rand(nn2, 1) - 0.5

    return w1, b1, w2, b2

def ReLU(x):
    return np.maximum(0, x)


# since we're dealing with batches, softmax is applied to each column of the input matrix
def softmax(input):
    output = np.exp(input - np.max(input, axis=0, keepdims=True))
    return output / np.sum(output, axis=0, keepdims=True)

def one_hot(Y, length):
    one_hot_Y = np.zeros((length, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


# dimensions:

# a0: (nn0, m) = (784, m)
# w1: (nn1, nn0) = (10, 784)
# b1: (nn1, 1) = (10, 1)
# z1: (nn1, m) = (10, m)

# a1: (nn1, m) = (10, m)
# w2: (nn2, nn1) = (10, 10)
# b2: (nn2, 1) = (10, 1)
# z2: (nn2, m) = (10, m)

# a2: (nn2, m) = (10, m)

def forward_propagation(w1, b1, w2, b2, a0):

    z1 = np.dot(w1, a0) + b1
    a1 = ReLU(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2) # we normalize z2 differently because we want the output to be a probability distribution

    return z1, a1, z2, a2

# backprop output:
# dw1: (nn1, nn0) = (10, 784)
# db1: (nn1, 1) = (10, 1)

# dw2: (nn2, nn1) = (10, 10)
# db2: (nn2, 1) = (10, 1)

def backward_propagation(w1, b1, w2, b2, a0, a1, a2, z1, z2, y, nn0, nn1, nn2):
    m = y.size  # no of training examples = no of columns in y

    # Compute gradients for the output layer
    dz2 = a2 - one_hot(y, nn2)
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    # Compute gradients for the hidden layer
    dz1 = np.dot(w2.T, dz2) * (z1 > 0)  # derivative of ReLU
    dw1 = (1/m) * np.dot(dz1, a0.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    return w1, b1, w2, b2

# no of columns in a0 = size of batch
# 1 backpropogation done = 1 batch training done = 1 update of weights and biases done
# if no of backpropogations done are such that all training examples are used, then 1 epoch is done
# if multiple epochs are done, then training is complete
    # we perform multiple epochs, starting with its already updated weights, and improves them further.
    # weights are not reset between epochs.

# this function performs 1 batch training
def gradient_descent_batch(a0, y, learning_rate, w1, b1, w2, b2, nn0, nn1, nn2):

    z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, a0)
    dw1, db1, dw2, db2 = backward_propagation(w1, b1, w2, b2, a0, a1, a2, z1, z2, y, nn0, nn1, nn2)
    w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

    return w1, b1, w2, b2

# this function performs 1 epoch of training
def gradient_descent_epoch(train_input, train_output, learning_rate, batch_size, w1, b1, w2, b2, nn0, nn1, nn2, epoch):

    total_no_of_training_examples = train_input.shape[1]  # number of training examples
    
    for i in range(0, total_no_of_training_examples, batch_size):
        print(f"epoch {epoch} | batch {i // batch_size}")
        a0 = train_input[:, i:i+batch_size]
        y = train_output[i:i+batch_size]
        w1, b1, w2, b2 = gradient_descent_batch(a0, y, learning_rate, w1, b1, w2, b2, nn0, nn1, nn2)

    return w1, b1, w2, b2


def train_model(train_input, train_output, learning_rate, batch_size, epochs, nn0, nn1, nn2):

    w1, b1, w2, b2 = initialize_parameters(nn0, nn1, nn2)  # Initialize only once

    for epoch in range(epochs):
        w1, b1, w2, b2 = gradient_descent_epoch(
            train_input, train_output, learning_rate, batch_size, 
            w1, b1, w2, b2, 
            nn0, nn1, nn2, epoch
        )
    return w1, b1, w2, b2


