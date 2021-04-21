import numpy as np
from Utils.utils_torch import Hook
import torch
from Layers.Conv2d import Conv2d


class Dropout:
    def __init__(self, probability_drop):
        """
        Dropout layer
        :param probability_drop: probability that the neuron will reset to zero
        """
        self.prob_drop = probability_drop
        self.latest_mask = None

    def forward(self, input, training):
        """
        Forward pass.
        :param input: input tensor
        :param training: when the model evaluate, the neurons should not be thrown out
        :return: output tensor with drop neurons
        """
        if training:
            mask = np.random.rand(*input.shape) > self.prob_drop
            output = (input * mask)/(1-self.prob_drop)
            self.latest_mask = mask
        else:
            output = input
        return output

    def backward(self, input, grad_output):
        """
        Backward pass. df/dx = df/dy * dy/dx, df/dw = df/dy * dy/dw, df/db = df/dy * dy/b
        :param input: not used
        :param grad_output: df/dy
        :return: grad_input (df/dx), df/dw: None
        """
        # we calculate the derivative only for those neurons that were used in training
        grad_input = (grad_output*self.latest_mask)/(1-self.prob_drop)
        return grad_input, None, None

    def update_weights(self, grad):
        return None


if __name__ == '__main__':
    torch_sum = 0
    my_sum = 0
    torch_sum_backward = 0
    my_sum_backward = 0
    for i in range(1000):

        weights1 = np.random.randn(2, 3, 2, 2)
        biases1 = np.random.randn(2)

        conv1 = torch.nn.Conv2d(3, 2, (2, 2))
        conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
        conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))

        dropout = torch.nn.Dropout(p=0.5)

        hook = Hook(dropout, backward=True)

        my_conv1 = Conv2d(3, 2, 2)
        my_conv1.weights = weights1.copy()
        my_conv1.biases = biases1.copy()
        my_dropout = Dropout(0.5)

        img = np.random.rand(7, 3, 15, 15)
        out = np.random.rand(7, 2, 14, 14)


        torch_res = dropout(conv1(torch.Tensor(img)))
        torch_sum += torch_res.sum()


        s1 = (torch_res - torch.Tensor(out)).sum()

        s1.backward()
        d_z = hook.output[0]

        torch_sum_backward += hook.input[0].sum()

        temp = my_conv1.forward(img)
        my_res = my_dropout.forward(temp,True)
        my_sum += my_res.sum()
        d = my_dropout.backward(temp, d_z.numpy())
        my_sum_backward += d[0].sum()
    print('NumPy')
    print('Result',my_sum/1000)
    print('dx',my_sum_backward/1000)
    print('Torch')
    print('Result',torch_sum/1000)
    print('dx',torch_sum_backward/1000)

