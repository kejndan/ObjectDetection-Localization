import numpy as np
import torch
from Utils.utils_torch import Hook


class Conv2d:
    def __init__(self, nb_in_channels, nb_out_channels, kernel_size, stride=1, add_padding=False):
        """
        Convolution layer 2D.
        :param nb_in_channels: number of input channels
        :param nb_out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride length
        :param add_padding: add padding or not
        """
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.add_padding = add_padding
        self.size_padding = 0

        self.weights = None
        self.biases = None

        self.grad_weights = None
        self.grad_biases = None
        self.__initialize_layer()

    def __initialize_layer(self):
        """
        This function initializes the weights and defines the padding size
        """
        self.weights = np.random.randn(self.nb_out_channels, self.nb_in_channels, self.kernel_size, self.kernel_size)
        self.biases = np.random.randn(1, self.nb_out_channels, 1, 1)

        if self.add_padding:
            self.size_padding = int((self.kernel_size - 1)/2)

    def __calc_output_dim(self, image_shape):
        """
        This function determines the size of the output tensor
        :param image_shape: input image shape
        :return: height, width
        """
        h_out = int((image_shape[1] - self.kernel_size + 2 * self.size_padding)/self.stride + 1)
        w_out = int((image_shape[2] - self.kernel_size + 2 * self.size_padding)/self.stride + 1)
        return h_out, w_out

    def add_pad(self, input):
        """
        This function adds padding
        :param input: tensor to add padding
        """
        return np.pad(input, ((0, 0), (0, 0),(self.size_padding, self.size_padding),(self.size_padding, self.size_padding)), mode='constant')

    def delete_pad(self, input):
        """
        This function deletes padding
        :param input: tensor to delete padding
        :return:
        """
        return input[:, :,self.size_padding:-self.size_padding, self.size_padding:-self.size_padding]

    def forward(self, input):
        """
        Forward pass.
        :param input: input tensor size of [batch_size, nb_in_channels, height_in, width_in]
        :return: output tensor size of [batch_size, nb_out_channels, height_out, width_out]
        """
        batch_size = input.shape[0]
        image_shape = input.shape[1:]
        h_out, w_out = self.__calc_output_dim(image_shape)
        padded_batch = self.add_pad(input)

        output = np.zeros((batch_size, self.nb_out_channels,h_out, w_out))

        for i in range(h_out):
            vertical_start = i * self.stride
            vertical_end = vertical_start + self.kernel_size
            for j in range(w_out):
                horizontal_start = j * self.stride
                horizontal_end = horizontal_start + self.kernel_size
                # print(vertical_start, vertical_end, horizontal_start, horizontal_end)
                # get a slice of the butch size [batch_size, .. : .., .. : .., nb_channels, 1] and
                # multiply it by a set of filters with the size [1, kernel_size, kernel_size, nb_in_channels, nb_out_channels]
                # the total size [batch_size, kernel_size, kernel_size, nb_in_channels, nb_out_channels]
                output[:, :,i, j] = (padded_batch[:, np.newaxis,:,vertical_start:vertical_end, horizontal_start:horizontal_end] *
                                      self.weights[np.newaxis]).sum(axis=(2,3,4))

        output = output + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        return output

    def backward(self, input, grad_output):
        """
        Backward pass. df/dx = df/dy * dy/dx, df/dw = df/dy * dy/dw, df/db = df/dy * dy/b
        :param input: the tensor that was applied to the input at the forward pass [batch_size, nb_in_channels, height_in, width_in]
        :param grad_output: df/dy (derivatives on this layer)  [batch_size, nb_out_channels, height_out, width_out]

        :return: df/dx size [batch_size, nb_in_channels, height_in, width_in],
                df/dw size [nb_out_channels, nb_in_channels, kernel_size, kernel_size],
                df/db size [nb_out_channels]
        """


        if self.add_padding:
            padded_input = self.add_pad(input)
        else:
            padded_input = input

        grad_input = np.zeros(input.shape)
        if self.add_padding:
            padded_grad_input = self.add_pad(grad_input)
        else:
            padded_grad_input = grad_input

        # compute df/db = df/dy * dy/db = \sum_{i,j,k} df/dy_{i,j,k}*dy_{i,j,k}/db,
        # where the amount for i is the amount for batch,
        # where the amount for j is the amount for row image,
        # where the amount for i is the amount for column image
        grad_biases = grad_output.sum(axis=(0, 2, 3))

        grad_weights = np.zeros(self.weights.shape)

        for i in range(grad_output.shape[2]):
            vertical_start = i * self.stride
            vertical_end = vertical_start + self.kernel_size
            for j in range(grad_output.shape[3]):
                horizontal_start = j * self.stride
                horizontal_end = horizontal_start + self.kernel_size

                # compute df/dx = df/dy * dy/dx
                # to do this, use the matrices from grad_outputs, we take a slice by [batch_size, nb_out_channels, 1, 1, 1]
                # and multiply by the filters [1, nb_out_channels, nb_in_channels, kernel_size, kernel_size]
                padded_grad_input[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end] += \
                    (self.weights[np.newaxis]*grad_output[:, :,i:i+1, j:j+1, np.newaxis]).sum(axis=1)

                # compute df/dw = df/dy * dy/dw
                # to do this, use the matrices from input, we take a slice by [batch_size, 1, nb_in_channels, kernel_size, kernel_size] (it's dy/dw)
                # and multiply by the grad_output [batch_size, nb_out_channels, 1, 1, 1] (it's df/dy)'
                # [1, nb_out_channels, nb_in_channels, kernel_size, kernel_size]
                grad_weights += (padded_input[:,np.newaxis, :,vertical_start:vertical_end, horizontal_start:horizontal_end, ]\
                                *grad_output[:, :, i:i+1, j:j+1, np.newaxis]).sum(axis=0)


        if self.add_padding:
            grad_input = self.delete_pad(padded_grad_input)
        return grad_input, grad_weights, grad_biases

    def update_weights(self, grad):
        self.weights -= grad[0]
        self.biases -= grad[1]


if __name__ == '__main__':
    np.random.seed(0)

    weights1 = np.random.randn(2,3,2,2)
    biases1 = np.random.randn(2)
    weights2 = np.random.randn(5,2,3,3)
    biases2 = np.random.randn(5)


    out = np.random.rand(7,5,7,7)


    conv1 = torch.nn.Conv2d(3,2,(2,2))
    conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
    conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))


    conv2 = torch.nn.Conv2d(2,5,(3,3), padding=1,stride=2)
    conv2.weight = torch.nn.Parameter(torch.Tensor(weights2))
    conv2.bias = torch.nn.Parameter(torch.Tensor(biases2))


    hook = Hook(conv2,backward=True)

    my_conv1 = Conv2d(3, 2, 2)
    my_conv1.weights = weights1.copy()
    my_conv1.biases = biases1.copy()
    my_conv2 = Conv2d(2,5,3,add_padding=True,stride=2)
    my_conv2.weights = weights2.copy()
    my_conv2.biases = biases2.copy()

    img = np.random.rand(7,3,15,15)

    print('Torch')
    torch_res = conv2(conv1(torch.Tensor(img)))
    print('Result', torch_res.sum())

    s1 = (torch_res - torch.Tensor(out)).sum()
    s1.backward()
    d_z = hook.output[0]
    print(d_z.size())

    print('dx',hook.input[0].size(),hook.input[0].sum())
    print('dw',hook.input[1].size(),hook.input[1].sum())
    print('db', hook.input[2].size(), hook.input[2].sum())

    print('NumPy')
    temp = my_conv1.forward(img)
    my_res = my_conv2.forward(temp)
    print(my_res.sum())
    d = my_conv2.backward(temp, d_z.numpy())
    print('dx',d[0].shape,d[0].sum())
    print('dw',d[1].shape,d[1].sum())
    print('db', d[2].shape, d[2].sum())
















