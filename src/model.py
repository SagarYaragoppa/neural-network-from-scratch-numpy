# importing neccesary libraly 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#here we will initialise following parameter
#here we have chosen number from -0.5 to 0.5 because they are best for the activation function and neural networks start with small, random, zero-centered weights so learning is stable and neurons don’t behave identically
def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2
#here we check the initialised parameter 
# W1, B1, W2, B2 = initialize_parameters()
# print("W1:\n", W1)
# print("B1:\n", B1)
# print("W2:\n", W2)
# print("B2:\n", B2)

# if i want random value between 0-255
# Random float values between 0 and 255
# W = np.random.rand(10, 784) * 255
# if you want decimal values, you’re experimenting / visualizing you’ll scale later
#Random integer values between 0 and 255 (like pixels brightness)
# W = np.random.randint(0, 256, size=(10, 784))


# these are the activation 
# relu covert negative value inro 0 and keep positive as it is 
# relu is not the best choices, we can use leaky relu 
def ReLU(X):
    return np.maximum(X, 0)
#softmax is a comman outter layer activation and convert the output to relative probability
def softmax_calculator(Z):
    expZ = np.exp   (Z - np.max(Z, axis=0))
    return expZ / np.sum(expZ, axis=0)



#here we are calculating value of ewquired parameter accourding to the formula
def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax_calculator(Z2)
    return Z1, A1, Z2, A2

#this is the one hot converter
def one_hot_converter(Y): # takes input y
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) #craete a zero array of size m,10
    one_hot_Y[np.arange(Y.size), Y] = 1 # now convert the number into one hot form
    return one_hot_Y.T #return one hot in transpose

def compute_loss(A2, Y):
    one_hot_Y = one_hot_converter(Y)
    # small epsilon to avoid log(0)
    loss = -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / Y.size
    return loss


# 
def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
    m = Y.size
    one_hot_Y = one_hot_converter(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  
    return dW1, dB1, dW2, dB2


# updating parameter before it again goes in forward propagation
def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
  W1 = W1 - learning_rate * dW1
  B1 = B1 - learning_rate * dB1
  W2 = W2 - learning_rate * dW2
  B2 = B2 - learning_rate * dB2
  return W1, B1, W2, B2

def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size