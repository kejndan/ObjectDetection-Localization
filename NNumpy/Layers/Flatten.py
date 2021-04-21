import numpy as np
from Utils.utils_torch import Hook
import torch
from Layers.Conv2d import Conv2d


class Flatten:
    def __init__(self):
        """
        Flatten layer
        """

    def forward(self, input):
        """
        Forward pass.
        :param input: input tensor size [batch_size, d_1, d_2, ..., d_n]
        :return: output tensor size [batch_size, d_1*d_2*...*d_n]
        """
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, input, grad_output):
        """
        Backward pass. df/dx = df/dy * dy/dx, df/dw = df/dy * dy/dw, df/db = df/dy * dy/b
        :param input: the tensor that was applied to the input at the forward pass [batch_size, d_1, d_2, ..., d_n]
        :param grad_output: df/dy (tensor size [batch_size, d_1*d_2*...*d_n])
        :return: grad_input (df/dx) (tensor size [batch_size, d_1, d_2, ..., d_n]), df/dw: None
        """
        return grad_output.reshape(*input.shape), None, None

    def update_weights(self, grad):
        return None

if __name__ == '__main__':
    np.random.seed(0)

    weights1 = np.random.randn(2,3,2,2)
    biases1 = np.random.randn(2)




    conv1 = torch.nn.Conv2d(3,2,(2,2))
    conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
    conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))

    flatten = torch.nn.Flatten()


    hook = Hook(flatten,backward=True)

    my_conv1 = Conv2d(3, 2, 2)
    my_conv1.weights = weights1.copy()
    my_conv1.biases = biases1.copy()
    my_flatten = Flatten()

    img = np.random.rand(7,3,15,15)
    out = np.random.rand(7, 392)

    print('Torch')
    torch_res = flatten(conv1(torch.Tensor(img)))

    print('Result', torch_res.sum())
    s1 = (torch_res - torch.Tensor(out)).sum()
    s1.backward()
    d_z = hook.output[0]


    print('dx',hook.input[0].size(),hook.input[0].sum())


    print('NumPy')
    temp = my_conv1.forward(img)
    my_res = my_flatten.forward(temp)
    print('Result', my_res.sum())
    d = my_flatten.backward(temp, d_z.numpy())
    print('dx',d[0].shape,d[0].sum())

