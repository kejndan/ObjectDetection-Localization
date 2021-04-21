import numpy as np
from Utils.utils_torch import Hook
import torch
from Layers.Conv2d import Conv2d


class Pool:
    def __init__(self, kernel_size, stride=1, mode='max'):
        """
        Pooling layers 2D
        :param kernel_size: scope of the filter max/average
        :param stride: stride length
        :param mode: max/average pooling
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def __calc_output_dim(self, image_shape):
        """
        This function determines the size of the output tensor
        :param image_shape: input image shape
        :return: height, width
        """
        h_out = int((image_shape[1] - self.kernel_size)/ self.stride + 1)
        w_out = int((image_shape[2] - self.kernel_size)/ self.stride + 1)
        return h_out, w_out

    def forward(self, input):
        """
        Forward pass.
        :param input: input tensor size of [batch_size, nb_in_channels, height_in, width_in]
        :return: output tensor size of [batch_size, nb_in_channels, height_out, width_out]
        """
        batch_size = input.shape[0]
        image_shape = input.shape[1:]
        h_out, w_out = self.__calc_output_dim(image_shape)
        output = np.zeros((batch_size, image_shape[0], h_out, w_out))

        for i in range(h_out):
            vertical_start = i * self.stride
            vertical_end = vertical_start + self.kernel_size
            for j in range(w_out):
                horizontal_start = j * self.stride
                horizontal_end = horizontal_start + self.kernel_size

                pool_function = None
                if self.mode == 'max':
                    pool_function = np.max
                elif self.mode == 'average':
                    pool_function = np.mean
                # we take the maximum/average from the filter area
                output[:, :, i, j] = pool_function(
                    input[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end], axis=(2, 3)
                )

        return output

    def backward(self, input, grad_output):
        """
        Backward pass. df/dx = df/dy * dy/dx, df/dw = df/dy * dy/dw, df/db = df/dy * dy/b
        :param input: the tensor that was applied to the input at the forward pass [batch_size, nb_in_channels, height_in, width_in]
        :param grad_output: df/dy (derivatives on this layer)  [batch_size, nb_in_channels, height_out, width_out]

        :return: df/dx size [batch_size, nb_in_channels, height_in, width_in],
                df/dw None
        """
        grad_input = np.zeros(input.shape)
        image_shape = input.shape[1:]
        h_out, w_out = self.__calc_output_dim(image_shape)

        for i in range(h_out):
            vertical_start = i * self.stride
            vertical_end = vertical_start + self.kernel_size
            for j in range(w_out):
                horizontal_start = j * self.stride
                horizontal_end = horizontal_start + self.kernel_size

                if self.mode == 'max':
                    # compute df/dx = df/dy * dy/dx
                    # from grad_output we take a slice [batch_size, nb_in_channels, 1, 1] (it's df/dy)
                    # and multiply by the mask [batch_size, nb_in_channels, kernel_size, kernel_size]
                    # where there are zeros everywhere, except for the cell
                    # that was the maximum when taking the maximum function
                    grad_input[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end] +=\
                        grad_output[:, :, i:i+1, j:j+1] *\
                        self.mask_with_argmax(input[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end])

                elif self.mode == 'average':
                    # compute df/dx = df/dy * dy/dx
                    # from grad_output we take a slice [batch_size, nb_in_channels, 1, 1] (it's df/dy)
                    # and divide by the number of elements involved in taking the average
                    grad_input[:, :, vertical_start:vertical_end, horizontal_start:horizontal_end] +=\
                        grad_output[:, :, i:i+1, j:j+1] / (self.kernel_size * self.kernel_size)

        return grad_input, None, None


    def mask_with_argmax(self, part_image):
        """
        Analogous to the armax function, but in space [batch_size, nb_in_channels]
        :param part_image: part image size [batch_size, nb_in_channels, kernel_size, kernel_size]
        :return: the part of the image where the unit is if there was a maximum value, otherwise zero
        """
        shapes_img = part_image.shape
        stretching = part_image.copy().reshape(shapes_img[0]*shapes_img[1], shapes_img[2]*shapes_img[3])
        arg_max_for_each_image = stretching.argmax(axis=1)
        stretching[:] = 0
        stretching[np.arange(stretching.shape[0]), arg_max_for_each_image] = 1
        compression = stretching.reshape(shapes_img[0], shapes_img[1], shapes_img[2], shapes_img[3])
        return compression

    def update_weights(self, grad):
        return None

if __name__ == '__main__':
    np.random.seed(0)

    weights1 = np.random.randn(2,3,2,2)
    biases1 = np.random.randn(2)




    conv1 = torch.nn.Conv2d(3,2,(2,2))
    conv1.weight = torch.nn.Parameter(torch.Tensor(weights1))
    conv1.bias = torch.nn.Parameter(torch.Tensor(biases1))

    pool = torch.nn.AvgPool2d((2,2),stride=3)


    hook = Hook(pool,backward=True)

    my_conv1 = Conv2d(3, 2, 2)
    my_conv1.weights = weights1.copy()
    my_conv1.biases = biases1.copy()
    my_pool = Pool(2,stride=3,mode='average')

    img = np.random.rand(7,3,15,15)
    out = np.random.rand(7, 2, 5, 5)

    print('Torch')
    torch_res = pool(conv1(torch.Tensor(img)))

    print('Result', torch_res.sum())

    s1 = (torch_res - torch.Tensor(out)).sum()
    s1.backward()
    d_z = hook.output[0]


    print('dx',hook.input[0].size(),hook.input[0].sum())


    print('NumPy')
    temp = my_conv1.forward(img)
    my_res = my_pool.forward(temp)
    print('Result', my_res.sum())
    d = my_pool.backward(temp, d_z.numpy())
    print('dx',d[0].shape,d[0].sum())

