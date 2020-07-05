import numpy as np

class ActivationFunction:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def sigmoid_derivates(x, steps):
        return (sigmoid(x+step) - sigmoid(x)) / step

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, hidden_nodes2, output_nodes, learning_rate=0.1):
        # Parameters
        self.input_shape = input_nodes
        self.hidden_shape = hidden_nodes
        self.hidden_shape2 = hidden_nodes2
        self.output_shape = output_nodes
        self.lr = learning_rate
        
        # Weights for each layer
        self.weights1 = np.random.normal(0.0, pow(self.input_shape, -0.5), (self.hidden_shape, self.input_shape))
        self.weights2 = np.random.normal(0.0, pow(self.hidden_shape, -0.5), (self.hidden_shape2, self.hidden_shape))
        self.weights3 = np.random.normal(0.0, pow(self.hidden_shape, -0.5), (self.output_shape, self.hidden_shape2))

##        # Other option
##        self.weights1 = np.random.randn(self.hidden_shape, self.input_shape) * 0.15
##        self.weights2 = np.random.randn(self.hidden_shape, self.hidden_shape) * 0.15
##        self.weights3 = np.random.randn(self.output_shape, self.hidden_shape) * 0.15

        # save accuracy through epochs of training and testing on testing set (in dataset_functions.py)
        self.accuracy = []

    def forward(self, inputs):
        # Hidden layer
        self.layer1_output = ActivationFunction.sigmoid(np.dot(self.weights1, inputs))  
        self.layer2_output = ActivationFunction.sigmoid(np.dot(self.weights2, self.layer1_output))
        # Output layer (final layer)
        self.layer3_output = ActivationFunction.sigmoid(np.dot(self.weights3, self.layer2_output))

    def back_propagation(self, inputs, targets):
        # output layer error is the (target - actual)
        output_errors = targets - self.layer3_output
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden neurons
        hidden_errors = np.dot(self.weights3.T, output_errors)
        hidden_errors_last = np.dot(self.weights2.T, hidden_errors)
        
        # update the weights for the links between the first and second (output) layers -> Cost function
        # Compute gradiant descent with back propagation
        d_weights3 = self.lr * np.dot((output_errors * self.layer3_output * (1.0 - self.layer3_output)), np.transpose(self.layer2_output))
        d_weights2 = self.lr * np.dot((hidden_errors * self.layer2_output * (1.0 - self.layer2_output)), np.transpose(self.layer1_output))
        d_weights1 = self.lr * np.dot((hidden_errors_last * self.layer1_output * (1.0 - self.layer1_output)), np.transpose(inputs))

        # update the weights with the slope of the cost (loss) function
        self.weights3 += d_weights3
        self.weights2 += d_weights2
        self.weights1 += d_weights1

    def train(self, inputs_array, target_array):
        # Make them 2D array
        inputs = np.array(inputs_array, ndmin=2).T
        targets = np.array(target_array, ndmin=2).T

        # Calculate signals between each layer and use activation function for output
        self.forward(inputs)

        # Calculate cost function/loss function (error) & gradiant descent and back propagate it through weights of each layer
        self.back_propagation(inputs, targets)

    def query(self, inputs_array):
        inputs = np.array(inputs_array, ndmin=2).T
        self.forward(inputs)
        return self.layer3_output
