import torch
import torch.nn as nn

# torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0,
# dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

# Input : (BS, C_in, T, H, W) // from dataloader : (BS, T, H, W, C) = (BS, 240, 224, 192, 3)

class C3D(nn.Module):
    def __init__(self, n_class, pretrained = False):
        super(C3D, self).__init__()

        # Input : BS, 3, 224, 228, 196 -> SLT
        # Input : BS, 3, 16, 224, 288  -> UCF
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 8, 224, 228, 196
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        # BS, 8, 224, 114, 98
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), padding=(1,0,0))
        # BS, 16, 224, 112, 96
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        # BS, 16, 112, 56, 48
        self.conv3a = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 64, 112, 56, 48
        self.conv3b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 64, 112, 56, 48                                                                           // Residual
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        # BS, 64, 112, 28, 24
        self.conv4a = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 112, 28, 24
        self.conv4b = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 112, 28, 24
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        # BS, 128, 56, 14, 12
        self.conv5a = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 56, 14, 12
        self.conv5b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 56, 14, 12                                                                           // Residual
        self.pool5 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        # BS, 128, 56, 7, 6
        self.fc6 = nn.Linear(128*7*6, 1660)
        self.fc7 = nn.Linear(1660, n_class)

        self.dropout = nn.Dropout(p = 0.5)
        self.lrelu   = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, X):
        #print(X.shape)
        X = X.permute(0,2,1,3,4)

        X = self.lrelu(self.conv1(X))
        X = self.pool1(X)

        X = self.lrelu(self.conv2(X))
        X = self.pool2(X)

        X = self.lrelu(self.conv3a(X))
        X = self.lrelu(self.conv3b(X)+X)
        X = self.pool3(X)

        X = self.lrelu(self.conv4a(X))
        X = self.lrelu(self.conv4b(X))
        X = self.pool4(X)
        #print("after pool4 : ", X.shape)

        X = self.lrelu(self.conv5a(X))
        X = self.lrelu(self.conv5b(X)+X)
        X = self.pool5(X)
        #print("after pool5 : ", X.shape)

        BS, T = X.shape[0], X.shape[2]
        X = X.view(BS, T, 128*7*6)
        #print("after reshape : ", X.shape)
        X = self.dropout(X)
        X = self.lrelu(self.fc6(X))
        X = self.dropout(X)
        logits = self.fc7(X)

        return logits

if __name__ == "__main__":
    inputs = torch.rand(8, 224, 228, 196, 3)
    net = C3D(n_class=512, pretrained=False)

    outputs = net.forward(inputs)
    print(outputs.shape)