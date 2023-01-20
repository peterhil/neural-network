import numpy as np

from numpy.random import randn


# X = (hours studying, hours sleeping), y = score on test
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float)  # input data
y = np.array(([92], [86], [89]), dtype=float)  # output

# scale units
x_all = x_all / np.max(x_all, axis=0)  # scaling input data
y = y / 100  # scaling output data (max test score is 100)

# split data
x = np.split(x_all, [3])[0]  # training data
x_predicted = np.split(x_all, [3])[1]  # testing data


def sigmoid(s):
    """activation function"""
    return 1 / (1 + np.exp(-s))


class NeuralNetwork:
    def __init__(self):
        """
        Initialise parameters
        """
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        # weights
        # (2x3) weight matrix from input to hidden layer
        self.w1 = randn(
            self.input_size,
            self.hidden_size,
        )
        # (3x1) weight matrix from hidden to output layer
        self.w2 = randn(
            self.hidden_size,
            self.output_size,
        )

    def forward(self, x):
        """
        forward propagation through our network
        """
        # dot product of x (input) and first set of 2x3 weights
        self.z = np.dot(
            x,
            self.w1,
        )
        self.z2 = sigmoid(self.z)  # activation function

        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(
            self.z2,
            self.w2,
        )
        out = sigmoid(self.z3)  # final activation function

        return out


nn = NeuralNetwork()

# defining our output
out = nn.forward(x)

print(f"Predicted Output: \n{ str(out) }")
print(f"Actual Output: \n{ str(y) }")
