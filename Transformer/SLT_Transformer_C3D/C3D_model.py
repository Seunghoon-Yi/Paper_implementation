import torch
import torch.nn as nn

# torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0,
# dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

# Input : (BS, C_in, T, H, W) // from dataloader : (BS, T, H, W, C) = (BS, 16, 240, 288, 3)

class C3D(nn.Module):
    def __init__(self, n_class, pretrained = False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 32, 16, 224, 288
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        # BS, 32, 16, 112, 144
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 64, 16, 112, 144
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        # BS, 64, 16, 56, 72
        self.conv3a = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 16, 56, 72
        self.conv3b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 128, 16, 56, 72
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        # BS, 128, 8, 28, 36
        self.conv4a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 256, 8, 28, 36
        self.conv4b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), padding=(1,1,1))
        # BS, 256, 8, 28, 36
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        # BS, 256, 4, 14, 18
        self.conv5a = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), padding=(1,0,0))
        # BS, 256, 4, 12, 16
        self.conv5b = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3,3,3), padding=(1,0,0))
        # BS, 128, 4, 10, 14
        self.pool5 = nn.AdaptiveMaxPool3d(output_size=(2, 4, 8))
        # BS, 128, 2, 4, 8
        self.fc6 = nn.Linear(8192, 1024)
        self.fc7 = nn.Linear(1024, n_class)

        self.dropout = nn.Dropout(p = 0.4)
        self.lrelu   = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, X):
        X = X.permute(0,4,1,2,3)

        X = self.lrelu(self.conv1(X))
        X = self.pool1(X)

        X = self.lrelu(self.conv2(X))
        X = self.pool2(X)

        X = self.lrelu(self.conv3a(X))
        X = self.lrelu(self.conv3b(X))
        X = self.pool3(X)

        X = self.lrelu(self.conv4a(X))
        X = self.lrelu(self.conv4b(X))
        X = self.pool4(X)
        #print(X.shape)

        X = self.lrelu(self.conv5a(X))
        X = self.lrelu(self.conv5b(X))
        X = self.pool5(X)
        #print(X.shape)

        X = X.view(-1, 8192)
        X = self.dropout(X)
        X = self.lrelu(self.fc6(X))
        X = self.dropout(X)
        logits = self.fc7(X)

        return logits

if __name__ == "__main__":
    inputs = torch.rand(8, 16, 240, 288, 3)
    net = C3D(n_class=101, pretrained=False)

    outputs = net.forward(inputs)
    print(outputs.shape)