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


def sigmoid_prime(s):
    """derivative of sigmoid"""
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self):
        """initialise parameters"""
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        # weights
        # (2x3) weight matrix from input to hidden layer
        self.w1 = randn(self.input_size, self.hidden_size)
        # (3x1) weight matrix from hidden to output layer
        self.w2 = randn(self.hidden_size, self.output_size)

    def forward(self, x):
        """forward propagation through our network"""
        # dot product of x (input) and first set of 2x3 weights
        self.z = np.dot(x, self.w1)
        # activation function
        self.z2 = sigmoid(self.z)

        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.w2)
        # final activation function
        out = sigmoid(self.z3)

        return out

    def backward(self, x, y, o):
        """backward propagate through the network"""
        # error in output
        self.o_error = y - o
        # applying derivative of sigmoid to error
        self.o_delta = self.o_error * sigmoid_prime(o)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.w2.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.w1 += x.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, x, y):
        """train the network with forward and backward propagation"""
        o = self.forward(x)
        self.backward(x, y, o)

    def predict(self):
        print("Predicted data based on trained weights:")
        print(f"Input (scaled): \n{ x_predicted }")
        print(f"Output: \n{ self.forward(x_predicted) }")

    def save_weights(self):
        np.savetxt("data/w1.txt", self.w1, fmt="%s")
        np.savetxt("data/w2.txt", self.w2, fmt="%s")


nn = NeuralNetwork()

print(f"Input:\n{ x }\n")
print(f"Actual Output:\n{ y }\n")

# train the nn 1,000 times
for i in range(1000):
    forward = nn.forward(x)
    mean_squared_error = np.mean(np.square(y - forward))

    print(f"Predicted Output:\n{ forward }")
    print(f"Loss: { mean_squared_error }")
    print()

    nn.train(x, y)

nn.save_weights()
nn.predict()
