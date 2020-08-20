import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        layer_size_conv = [48, 48, 96]
        layer_size_lear = [2048]
        
        ### First layer - convolution 1
        # (3, 224, 224) three channels (RGB) input image
        # Xavier weight initialization
        # 48 5x5 square convolution kernel with stride 2
        # 48 extracted output channels/feature maps
        # (48, 110, 110) output
        # Batch normalization
        self.conv1 = nn.Conv2d(3, layer_size_conv[0],
                               5, stride=2)
        I.xavier_normal_(self.conv1.weight, gain=1.)
        self.norm1 = nn.BatchNorm2d(layer_size_conv[0])
        
        ### Second layer - convolution 2
        # (48, 110, 110) input
        # Xavier weight initialization
        # 48 3x3 square convolution kernel with stride 2
        # 48 extracted output channels/feature maps
        # (48, 54, 54) output
        # Batch normalization
        self.conv2 = nn.Conv2d(layer_size_conv[0],
                               layer_size_conv[1],
                               3, stride=2)
        I.xavier_normal_(self.conv2.weight, gain=1.)
        self.norm2 = nn.BatchNorm2d(layer_size_conv[1])

        ### Third layer - convolution 3
        # (48, 54, 54) input
        # Xavier weight initialization
        # 48 3x3 square convolution kernel with stride 2
        # 48 extracted output channels/feature maps
        # (96, 26, 26) intermediary output
        # Batch normalization
        # (2, 2) Pooling
        # (96, 13, 13) output
        self.conv3 = nn.Conv2d(layer_size_conv[1],
                               layer_size_conv[2],
                               3, stride=2)
        I.xavier_normal_(self.conv3.weight, gain=1.)
        self.norm3 = nn.BatchNorm2d(layer_size_conv[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        ### Fourth layer - fully connected 1
        # 96 x 13 x 13 input size
        # Xavier weight initialization
        # Batch normalization
        # 30% dropout
        # 2048 output size
        self.fc1 = nn.Linear(layer_size_conv[2]*13*13,
                             layer_size_lear[0])
        I.xavier_normal_(self.fc1.weight, gain=1.)
        self.nf1 = nn.BatchNorm1d(layer_size_lear[0])
        self.df1 = nn.Dropout(p=0.3)
        
        ### Fifth/Output layer - fully connected 2
        # 2048 input size
        # 133 output size
        self.output = nn.Linear(layer_size_lear[-1], 133)
    
    def forward(self, x):
        # ReLU activation for all convolutional layers
        # and the first fully connected layer
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.df1(self.nf1(self.fc1(x))))
        x = self.output(x)
        
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        layer_size_lear = [2048]

        resnet = models.resnet50(pretrained=True)  # change it for other pretrained models
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
     
        self.fc1 = nn.Linear(resnet.fc.in_features, layer_size_lear[0])
        I.xavier_normal_(self.fc1.weight, gain=1.)
        self.nf1 = nn.BatchNorm1d(layer_size_lear[0])
        self.df1 = nn.Dropout(p=0.3)
               
        self.output = nn.Linear(layer_size_lear[-1], 133)

    def forward(self, x):
        ## Define forward behavior
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.df1(self.nf1(self.fc1(x))))
        x = self.output(x)

        return x
