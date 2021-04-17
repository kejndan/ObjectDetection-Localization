import torch


class LocalizationLoss(torch.nn.Module):
    def __init__(self, size_img=(220, 220)):
        super(LocalizationLoss, self).__init__()
        self.size_img = size_img

    def forward(self, output, target):
        width = self.size_img[0]
        height = self.size_img[1]

        l_x1 = torch.pow(output[:, 0] - target[:, 0] / width, 2)
        l_y1 = torch.pow(output[:, 1] - target[:, 1] / height, 2)
        l_x2 = torch.pow(output[:, 2] - target[:, 2] / width, 2)
        l_y2 = torch.pow(output[:, 3] - target[:, 3] / height, 2)
        return (l_x1 + l_y1 + l_x2 + l_y2).sum()
