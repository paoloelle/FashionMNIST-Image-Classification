import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(7 * 7 * 128, 10)

    def forward(self, x):
        conv1 = self.conv1(x)  # torch.Size([64, 32, 14, 14])
        conv2 = self.conv2(conv1)  # torch.Size([64, 64, 7, 7])
        conv3 = self.conv3(conv2)  # torch.Size([64, 128, 7, 7])
        out = conv3.view(conv3.size(0), -1)

        return self.fc(out)
