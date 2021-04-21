import numpy as np

class SGD:
    def __init__(self, model,learning_rate=1e-3):
        self.model = model
        self.learning_rate = learning_rate

    def update(self):
        for module in self.model:
            gradient_step = []
            if hasattr(module, 'grad_weights'):
                gradient_step.append(self.learning_rate*module.grad_weights)
            if hasattr(module, 'grad_biases'):
                gradient_step.append(self.learning_rate*module.grad_biases)