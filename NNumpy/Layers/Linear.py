import numpy as np
import torch
from NNumpy.Layers import Conv2d
from NNumpy.Layers import Flatten
from Utils.utils_torch import Hook


class Linear:
    def __init__(self, nb_in_features, nb_out_features):
        """
        Linear layer
        :param nb_in_features: number of input neurons
        :param nb_out_features: number of output neurons
        """
        self.nb_in_features = nb_in_features
        self.nb_out_features = nb_out_features

        self.weights = None
        self.biases = None

        self.grad_weights = None
        self.biases = None
        self.__initialize_layer()

    def __initialize_layer(self):
        """
        This function initializes the weights
        """
        self.weights = np.random.randn(self.nb_out_features,self.nb_in_features) * np.sqrt(2/self.nb_in_features)
        self.biases = np.zeros((1, self.nb_out_features))

    def forward(self, input):
        """
        Forward pass.
        :param input: input tensor size of [batch_size, nb_in_features]
        :return: output tensor size of [batch_size, nb_out_features]
        """
        return input @ self.weights.T + self.biases

    def backward(self, input, grad_output):
        """
        Backward pass. df/dx = df/dy * dy/dx, df/dw = df/dy * dy/dw, df/db = df/dy * dy/b
        :param input: the tensor that was applied to the input at the forward pass [batch_size, nb_in_features]
        :param grad_output: df/dy (derivatives on this layer)  [batch_size, nb_out_features]

        :return: df/dx size [batch_size, nb_in_features],
                df/dw size [nb_out_features, nb_in_features],
                df/db size [1,nb_out_channels]
        """

        grad_input = grad_output @ self.weights
        grad_weights = grad_output.T @ input
        grad_biases = grad_output.sum(axis=0)
        return grad_input, grad_weights, grad_biases

    def update_weights(self, grad):
        self.weights -= grad[0]
        self.biases -= grad[1]

if __name__ == '__main__':
    np.random.seed(0)

    weights1 = np.random.randn(2,3,2,2)
    biases1 = np.random.randn(2)

    weights_linear = np.random.randn(10, 392)
    biases_linear = np.random.randn(1,10)



    conv1 = torch.nn.Conv2d(3,2,(2,2))
    conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
    conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))

    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(392, 10)
    linear.weight = torch.nn.Parameter(torch.Tensor(weights_linear))
    linear.bias = torch.nn.Parameter(torch.Tensor(biases_linear))



    hook = Hook(linear,backward=True)

    my_conv1 = Conv2d(3, 2, 2)
    my_conv1.weights = weights1.copy()
    my_conv1.biases = biases1.copy()
    my_flatten = Flatten()
    my_linear = Linear(392,10)
    my_linear.weights = weights_linear.copy()
    my_linear.biases = biases_linear.copy()

    img = np.random.rand(7,3,15,15)
    out = np.random.rand(7, 10)

    print('Torch')
    k = flatten(conv1(torch.Tensor(img)))
    torch_res = linear(k)

    print('Result', torch_res.sum())
    s1 = (torch_res - torch.Tensor(out)).sum()
    s1.backward()
    d_z = hook.output[0]


    print('dx',hook.input[1].size(),hook.input[1].sum())
    print('dw',hook.input[2].size(),hook.input[2].sum())
    print('db', hook.input[0].size(), hook.input[0].sum())


    print('NumPy')
    temp = my_conv1.forward(img)
    temp = my_flatten.forward(temp)
    my_res = my_linear.forward(temp)
    print('Result', my_res.sum())
    d = my_linear.backward(temp, d_z.numpy())
    print('dx',d[0].shape,d[0].sum())
    print('dw',d[1].shape,d[1].sum())
    print('db', d[2].shape, d[2].sum())

