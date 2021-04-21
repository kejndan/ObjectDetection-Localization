from torchvision.models import mobilenet_v3_small
from torchsummary import summary
import torch


class ConvNet_w_MN_1(torch.nn.Module):
    def __init__(self):
        super(ConvNet_w_MN_1, self).__init__()
        mobile_net = mobilenet_v3_small(pretrained=True)
        self.features = mobile_net.features
        self.avgpool = mobile_net.avgpool
        self.linear_localize = torch.nn.Linear(576, 4)
        self.linear_classify = torch.nn.Linear(576, 1)

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(input.size(0),-1)
        coords = self.linear_localize(x)
        classes = self.linear_classify(x)
        return torch.cat((classes, coords),axis=1)

if __name__ == '__main__':
    pass