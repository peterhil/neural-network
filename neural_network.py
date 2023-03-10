#!/usr/bin/env python
# Usage: python -i neural_network.py -p

"""
Simple neural network with backpropagation.
https://enlight.nyc/projects/neural-network
"""

import click
import sys

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import randn


def sigmoid(s):
    """activation function"""
    return 1 / (1 + np.exp(-s))


def sigmoid_prime(s):
    """derivative of sigmoid"""
    return s * (1 - s)


def plot(count, losses, outputs):
    """plot the iterations in real time
    """
    plt.cla()
    plt.title("Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(count, losses)
    plt.plot(count, 1 - np.array(losses))
    plt.plot(count, outputs)
    plt.pause(.001)


class NeuralNetwork:
    """Simple neural network with backpropagation.
    """
    def __init__(self):
        """Initialise network with parameters"""
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        # weights
        # (2x3) weight matrix from input to hidden layer
        self.weigths_input = randn(self.input_size, self.hidden_size)
        # (3x1) weight matrix from hidden to output layer
        self.weigths_hidden = randn(self.hidden_size, self.output_size)

    def forward(self, inputs):
        """Forward propagation through our network"""
        # dot product of x (input) and first set of 2x3 weights
        self.z = np.dot(inputs, self.weigths_input)
        # activation function
        self.z2 = sigmoid(self.z)

        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.weigths_hidden)
        # final activation function
        outputs = sigmoid(self.z3)

        return outputs

    def backward(self, inputs, targets, outputs):
        """Backward propagate through the network"""
        # error in output
        self.o_error = targets - outputs
        # applying derivative of sigmoid to error
        self.o_delta = self.o_error * sigmoid_prime(outputs)

        # how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.weigths_hidden.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.weigths_input += inputs.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.weigths_hidden += self.z2.T.dot(self.o_delta)

    def train(self, inputs, targets):
        """Train the network with forward and backward propagation"""
        outputs = self.forward(inputs)
        self.backward(inputs, targets, outputs)

    def predict(self, inputs):
        """Make a prediction based on trained weights"""
        print("Predicted data based on trained weights:")
        print(f"Input (scaled): \n{ inputs }")
        print(f"Output: \n{ self.forward(inputs) }")

    def save_weights(self):
        """Save weigths into data directory"""
        np.savetxt("data/weigths_input.txt", self.weigths_input, fmt="%s")
        np.savetxt("data/weigths_hidden.txt", self.weigths_hidden, fmt="%s")


def train_network(network, inputs, targets, n=1000, limit=0.99, show_plot=False, verbose=False):
    """Train the network in a loop"""
    counts = []  # list to store iteration count
    losses = []  # list to store loss values
    mean_outputs = []

    # train the network 1,000 times
    for i in range(n):
        forward = network.forward(inputs)
        loss = np.mean(np.square(targets - forward))  # mean squared error

        if verbose:
            print(f"Predicted Output:\n{ forward }")
            print(f"Loss: { loss }\n")

        counts.append(i)
        losses.append(np.round(float(loss), 6))
        mean_outputs.append(np.mean(forward))

        if show_plot:
            plot(counts, losses, mean_outputs)

        if loss <= limit:
            print(f'Limit { limit * 100 }% at count: {i}')
            break

        network.train(inputs, targets)


@click.command()
@click.option('--limit', '-l', default=0.001, help='Break early when loss equals limit')
@click.option('--number', '-n', default=1000, help='Number of iterations to train the network')
@click.option('--plot', '-p', is_flag=True, help='Plot progress with Matplotlib')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(number, limit=0.001, plot=False, verbose=False):
    # X = (hours studying, hours sleeping), y = score on test
    x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float)  # input data
    y = np.array(([92], [86], [89]), dtype=float)  # output

    # scale units
    x_all = x_all / np.max(x_all, axis=0)  # scaling input data
    y = y / 100  # scaling output data (max test score is 100)

    # split data
    x = np.split(x_all, [3])[0]  # training data
    x_predicted = np.split(x_all, [3])[1]  # testing data

    print(f"Input:\n{ x }\n")
    print(f"Actual Output:\n{ y }\n")

    nn = NeuralNetwork()

    train_network(nn, x, y, limit=limit, show_plot=plot, verbose=verbose)
    nn.save_weights()
    nn.predict(x_predicted)


if __name__ == '__main__':
    main()
