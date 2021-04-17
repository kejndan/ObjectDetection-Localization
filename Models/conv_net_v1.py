from torch import nn


class ConvNetV1(nn.Module):
    def __init__(self, nb_outs=5):
        super(ConvNetV1, self).__init__()
        self.nb_outs = nb_outs

        self.conv1 = nn.Conv2d(3, 32, 21)
        self.conv2 = nn.Conv2d(32, 64, 11)
        self.conv3 = nn.Conv2d(64, 128, 6)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.dense1 = nn.Linear(16384, 1024)
        self.dense2 = nn.Linear(1024, 5)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        x = self.pool(self.conv1(input))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


