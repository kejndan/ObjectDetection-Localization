from torch import nn


class ConvNetV3(nn.Module):
    def __init__(self, nb_outs=5):
        super(ConvNetV3, self).__init__()
        self.nb_outs = nb_outs

        self.conv1 = nn.Conv2d(3, 32, 21)
        self.conv2 = nn.Conv2d(32, 64, 11)
        self.conv3 = nn.Conv2d(64, 128, 6)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.dense1 = nn.Linear(16384, 1024)
        self.dense2 = nn.Linear(1024, 5)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout_p_0_5 = nn.Dropout(0.5)
        self.dropout_p_0_2 = nn.Dropout(0.2)

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.dropout_p_0_2(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x - self.dropout_p_0_2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.dropout_p_0_2(x)
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.dropout_p_0_2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout_p_0_5(x)
        x = self.dense2(x)
        return x


