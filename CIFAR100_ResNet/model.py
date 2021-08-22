import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conv3x3 block
def conv3x3(in_ch, out_ch, stride = 1) ->nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
# Conv1x1 block
def conv1x1(in_ch, out_ch, stride = 1) ->nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

# Basic Residual block for ResNet 18 / 34
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample = None):
        super(ResBlock, self).__init__()
        # Sequential(residual fcn)
        self.residualBlock = nn.Sequential(
            conv3x3(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch)
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out      = self.residualBlock(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out     += identity
        out     = nn.ReLU(inplace=True)(out)

        return out

# Bottleneck block for ResNet 50 / 101 / 152
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride = 1, downsample = None):
        super(BottleNeck, self).__init__()
        self.residualBlock = nn.Sequential(
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            conv3x3(out_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            conv1x1(out_ch, out_ch*BottleNeck.expansion),
            nn.BatchNorm2d(out_ch*BottleNeck.expansion)
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out      = self.residualBlock(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        #print(x.shape, out.shape, identity.shape)
        out     += identity
        out     =  nn.ReLU(inplace=True)(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, n_layers, n_class = 100):
        super(ResNet, self).__init__()
        # The initial Conv layer
        self.in_ch  = 16
        self.conv   = conv3x3(3,16)
        self.BN     = nn.BatchNorm2d(16)
        self.relu   = nn.ReLU(inplace=True)
        # Blocks.
        self.layer1 = self.make_layer(block, 32, n_layers[0])
        self.layer2 = self.make_layer(block, 64, n_layers[1], stride=2)
        self.layer3 = self.make_layer(block, 128, n_layers[2], stride=2)
        self.layer4 = self.make_layer(block, 256, n_layers[3], stride=2)
        # Output layer elements
        self.pool   = nn.AdaptiveAvgPool2d((1,1))                     # The output size.
        self.fc1    = nn.Linear(256*block.expansion, n_class)
        self.drop   = nn.Dropout(p=0.5)

    def make_layer(self, block, out_ch, n_block, stride = 1):
        '''
        :param block:   block type, ResBlock or BottleNeck
        :param out_ch:  numbers of output channels
        :param n_block: number of blocks that will be stacked in the layer
        :param stride:  stride of the initial block of the layer.
        :return:        The entire ResNet model.
        '''
        downsample = None
        # Usage : for example, output of layer 1 shape = ( , 16, 32, 32)
        #         1) Here self.in_ch = 16
        #         2) out_ch of layer 2 = 32, hence downsample will be
        #            conv3x3(in = 16, out = 32, stride = 2) and added in the first block of the layer.
        if (stride!=1) or (self.in_ch!=out_ch):
            downsample = nn.Sequential(
                conv3x3(self.in_ch, out_ch*block.expansion, stride=stride),
                nn.BatchNorm2d(out_ch*block.expansion)
            )

        layers = []
        layers.append(block(self.in_ch, out_ch, stride, downsample)) # Add the block with downsample layer!

        self.in_ch = out_ch*block.expansion
        # Repeat blocks with the given list of numbers :
        for i in range(n_block-1):
            layers.append(block(self.in_ch, out_ch))
            #print(f'block {i+1} in/out : ', self.in_ch, out_ch)
            self.in_ch = out_ch*block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.BN(out)
        out = self.relu(out)
        # Body
        #print("after input layer : ", out.shape)
        out = self.layer1(out)
        #print("after layer 1 : ", out.shape)
        out = self.layer2(out)
        #print("after layer 2 : ", out.shape)
        out = self.layer3(out)
        #print("after layer 3 : ", out.shape)
        out = self.layer4(out)
        #print("after layer 4 : ", out.shape)
        # Output
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.fc1(out)

        return out

def ResNet18():
    return ResNet(ResBlock, [2,2,2,2])

def ResNet34():
    return ResNet(ResBlock, [3,4,6,3])

def ResNet50():
    return ResNet(BottleNeck, [3,4,6,3])

def ResNet101():
    return ResNet(BottleNeck, [3,4,23,3])

def ResNet152():
    return ResNet(BottleNeck, [3,8,36,3])

test_X = torch.randn(12, 3, 32, 32).to(device)
model = ResNet50()
model = model.to(device)

out = model(test_X)
print(out.shape)
summary(model, (3,32,32), device = device.type)

