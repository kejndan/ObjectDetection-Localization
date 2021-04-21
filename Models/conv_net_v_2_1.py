from torch import nn
import torch

class ConvNetV2_1(nn.Module):
    def __init__(self, nb_outs=5):
        super(ConvNetV2_1, self).__init__()
        self.nb_outs = nb_outs

        self.conv1 = nn.Conv2d(3, 8, 41, stride=2) # 90
        self.conv2 = nn.Conv2d(8, 8, 19, stride=2) # 36
        self.conv3 = nn.Conv2d(8, 16, 3) # 16
        self.conv4 = nn.Conv2d(16, 16, 3) # 14
        self.dense1 = nn.Linear(784, 512)
        self.linear_localize = nn.Linear(512, 4)
        self.linear_classify = nn.Linear(512, 1)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout_p_0_5 = nn.Dropout(0.5)
        self.dropout_p_0_2 = nn.Dropout(0.2)

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_p_0_2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout_p_0_2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dropout_p_0_5(x)
        coords = self.linear_localize(x)
        classes = self.linear_classify(x)
        return torch.cat((classes, coords),axis=1)


