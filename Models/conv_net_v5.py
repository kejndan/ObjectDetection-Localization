from torch import nn


class ConvNetV5(nn.Module):
    def __init__(self, nb_outs=5):
        super(ConvNetV5, self).__init__()
        self.nb_outs = nb_outs

        self.conv1 = nn.Conv2d(3, 16, 21)
        self.conv2 = nn.Conv2d(16, 32, 11)
        self.conv3 = nn.Conv2d(32, 64, 6)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.conv5 = nn.Conv2d(128, 256, 5)
        # self.dense1 = nn.Linear(4096, 1024)
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
        x = self.relu(self.conv5(x))
        x = self.dropout_p_0_2(x)
        x = self.pool(x)
        x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dropout_p_0_5(x)
        x = self.dense2(x)
        return x


