import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.loss = None
        self.loss_function = None
        self.target = None

    def add_module(self, module):
        self.layers.append(module)

    def add_loss_function(self, function):
        self.loss_function = function

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, input):
        self.activations.append(input)
        for layer in self.layers:
            input = layer.forward(input)
            self.activations.append(input)
        return input

    def loss(self, prediction, target):
        if isinstance(self.loss_function, list):
            loss_val = 0
            last_index = 0
            for loss in self.loss_function:
                loss_val += loss(prediction[:,last_index:loss.nb_inputs], target[:,last_index:loss.nb_inputs])
                last_index += loss.nb_inputs
            self.loss = loss_val
        else:
            self.loss = self.loss_function(prediction, target)
        self.target = target
        return self.loss

    def backward(self):
        if isinstance(self.loss_function, list):
            grad_output = np.zeros(self.activations[-1].shape)
            last_index = 0
            for loss in self.loss_function:
                grad_output[:, last_index:loss.nb_inputs] += self.loss.backward(self.activations[-1][:,last_index:loss.nb_inputs],
                                                  self.target[:,last_index:loss.nb_inputs])[0]
                last_index += loss.nb_inputs
        else:
            grad_output = self.loss_function.backward(self.activations[-1], self.target)[0]
        for i, layer in enumerate(self.layers[::-1]):
            grad_output = self.layer.backward(self.activations[-i-2], grad_output)[0]

    def optimize(self):
        self.optimizer.update()


