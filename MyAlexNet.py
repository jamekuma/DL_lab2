import torch
import torch.nn as nn
import torchvision
import torchvision.models.alexnet


class MyAlexNet(torch.nn.Module):

    def __init__(self, n_classes=10):
        super(MyAlexNet, self).__init__()
        self.conv_layers = nn.Sequential(  # 五层卷积层
            # 卷积层1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  #
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.LocalResponseNorm(size=5),
            # 卷积层2
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.LocalResponseNorm(size=5),
            # 卷积层3
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 卷积层4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 卷积层5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # 全连接层1
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4*4*256, out_features=4096),
            nn.ReLU(inplace=True),
            # 全连接层2
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            # 全连接层3
            nn.Linear(in_features=4096, out_features=n_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 4*4*256)
        x = self.fc_layers(x)
        return x




