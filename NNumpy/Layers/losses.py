import numpy as np


class LocalizationLoss:
    def __init__(self, size_img=(220, 220)):
        super(LocalizationLoss, self).__init__()
        self.width = size_img[0]
        self.height = size_img[1]
        self.nb_inputs = 4

    def forward(self, output, target):
        l_x1 = np.power(output[:, 0] - target[:, 0] / self.width, 2)
        l_y1 = np.power(output[:, 1] - target[:, 1] / self.height, 2)
        l_x2 = np.power(output[:, 2] - target[:, 2] / self.width, 2)
        l_y2 = np.power(output[:, 3] - target[:, 3] / self.height, 2)
        return (l_x1 + l_y1 + l_x2 + l_y2).sum()

    def backward(self, output, target):
        grad_input = np.zeros(output.shape)
        grad_input += 2*(output[:, 0] - target[:, 0] / self.width)
        grad_input += 2 * (output[:, 1] - target[:, 1] / self.height)
        grad_input += 2 * (output[:, 2] - target[:, 2] / self.width)
        grad_input += 2 * (output[:, 3] - target[:, 3] / self.height)
        return grad_input, None, None


class BCEWithLogitsLoss:
    def __init__(self):
        self.nb_inputs = 1

    def forward(self, output, target):
        return (target*np.log(1/(1+np.exp(-output))) + (1 - target)*np.log(1 - 1/(1+np.exp(-output)))).mean()

    def backward(self, output, target):
        grad_input = np.zeros(output.shape)
        grad_input += 2/output.shape[0]*(target*output - output)
        return grad_input, None, None




