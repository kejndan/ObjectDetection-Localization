import numpy as np
import torch
from NNumpy.Layers import Conv2d
from NNumpy.Layers import Flatten
from Utils.utils_torch import Hook
from NNumpy.Layers import Linear


class ReLU:
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        grad_input = input > 0
        return grad_output*grad_input, None, None


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        return 1/(1 + np.exp(-input))

    def backward(self, input, grad_output):
        return grad_output * np.exp(-input)/np.power(1 + np.exp(-input),2), None, None


if __name__ == '__main__':
    np.random.seed(0)

    weights1 = np.random.randn(2,3,2,2)
    biases1 = np.random.randn(2)

    weights_linear = np.random.randn(10, 392)
    biases_linear = np.random.randn(1,10)



    conv1 = torch.nn.Conv2d(3,2,(2,2))
    conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
    conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))
    relu = torch.nn.ReLU()
    sigmoid = torch.nn.Sigmoid()
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(392, 10)
    linear.weight = torch.nn.Parameter(torch.Tensor(weights_linear))
    linear.bias = torch.nn.Parameter(torch.Tensor(biases_linear))



    hook_relu = Hook(relu,backward=True)
    hook_sigmoid = Hook(sigmoid,backward=True)

    my_conv1 = Conv2d(3, 2, 2)
    my_conv1.weights = weights1.copy()
    my_conv1.biases = biases1.copy()
    my_relu = ReLU()
    my_flatten = Flatten()
    my_linear = Linear(392,10)
    my_sigmoid = Sigmoid()
    my_linear.weights = weights_linear.copy()
    my_linear.biases = biases_linear.copy()

    img = np.random.rand(7,3,15,15)
    out = np.random.rand(7, 10)

    print('Torch')
    k = flatten(relu(conv1(torch.Tensor(img))))
    torch_res = sigmoid(linear(k))

    print('Result', torch_res.sum())
    s1 = (torch_res - torch.Tensor(out)).sum()
    s1.backward()
    d_z = hook_sigmoid.output[0]


    print('dx ReLU',hook_relu.input[0].size(),hook_relu.input[0].sum())
    print('dx Sigmoid', hook_sigmoid.input[0].size(), hook_sigmoid.input[0].sum())



    print('NumPy')
    temp1 = my_conv1.forward(img)
    temp2 = my_relu.forward(temp1)
    temp3 = my_flatten.forward(temp2)
    temp4 = my_linear.forward(temp3)
    my_res = my_sigmoid.forward(temp4)
    print('Result', my_res.sum())
    d_sigmoid = my_sigmoid.backward(temp4, d_z.numpy())
    print('dx Sigmoid',d_sigmoid[0].shape,d_sigmoid[0].sum())
    d_relu = my_relu.backward(temp1,my_flatten.backward(temp2,my_linear.backward(temp3,d_sigmoid[0])[0])[0])
    print('dx ReLU',d_relu[0].shape,d_relu[0].sum())
