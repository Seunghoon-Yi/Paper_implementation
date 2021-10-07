import torch.nn as nn
import torch
import os


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1.e-4, momentum=0.01, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                              stride=(1, stride, stride), padding=(0, padding, padding))
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                               padding=(padding, 0, 0))

        self.bn = nn.BatchNorm3d(out_planes, eps=1.e-4, momentum=0.01, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.bn2 = nn.BatchNorm3d(out_planes, eps=1.e-4, momentum=0.01, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        #nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        #nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        # x=self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


# Note the operations here for S3D-G:
# If we set two convs: 1xkxk + kx1x1, it's as follows: (p=(k-1)/2)
# BasicConv3d(input,output,kernel_size=(1,k,k),stride=1,padding=(0,p,p))
# Then BasicConv3d(output,output,kernel_size=(k,1,1),stride=1,padding=(p,0,0))

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(64, 32, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(64, 32, kernel_size=1, stride=1),
            STConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(64, 12, kernel_size=1, stride=1),
            STConv3d(12, 16, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(64, 16, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(128, 24, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(128, 48, kernel_size=1, stride=1),
            STConv3d(48, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(128, 16, kernel_size=1, stride=1),
            STConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(128, 24, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1) + x
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(128, 96, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(128, 48, kernel_size=1, stride=1),
            STConv3d(48, 104, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(128, 8, kernel_size=1, stride=1),
            STConv3d(8, 24, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(128, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(256, 80, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 56, kernel_size=1, stride=1),
            STConv3d(56, 112, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 16, kernel_size=1, stride=1),
            STConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            STConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            STConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            STConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            STConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(256, 64, kernel_size=1, stride=1),          # 4
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 72, kernel_size=1, stride=1),
            STConv3d(72, 128, kernel_size=3, stride=1, padding=1),  # 6
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 16, kernel_size=1, stride=1),
            STConv3d(16, 32, kernel_size=3, stride=1, padding=1),   # 3
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 32, kernel_size=1, stride=1),          # 3
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1) + x
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),           # 4
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 40, kernel_size=1, stride=1),
            STConv3d(40, 48, kernel_size=3, stride=1, padding=1),    # 6
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 12, kernel_size=1, stride=1),
            STConv3d(12, 24, kernel_size=3, stride=1, padding=1),     # 3
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 24, kernel_size=1, stride=1),           # 3
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(128, 48, kernel_size=1, stride=1),          # 3
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(128, 24, kernel_size=1, stride=1),
            STConv3d(24, 48, kernel_size=3, stride=1, padding=1),  # 3
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(128, 12, kernel_size=1, stride=1),
            STConv3d(12, 16, kernel_size=3, stride=1, padding=1),    # 1
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(128, 16, kernel_size=1, stride=1),           # 1
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1) + x
        return out


class S3DG(nn.Module):

    def __init__(self, embed_size = 512, device = None, dropout = 0.5, input_channel=3, spatial_squeeze=True):
        super(S3DG, self).__init__()
        self.features = nn.Sequential(
            STConv3d(input_channel, 32, kernel_size=6, stride=2, padding=2),           # (32, 120, 112, 96)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (32, 120, 56, 48)
            STConv3d(32, 64, kernel_size=3, stride=1, padding=1),                      # (64, 120, 56, 48)
            BasicConv3d(64, 64, kernel_size=1, stride=1),                              # (64, 120, 56, 48)
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),  # (64,  60, 28, 24)
            Mixed_3b(),                                                                # (128, 60, 28, 24)
            Mixed_3c(),                                                                # (128, 60, 28, 24) // residual
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),  # (128, 60, 14, 12)
            Mixed_4b(),                                                                # (256, 60, 14, 12)
            Mixed_4f(),                                                                # (256, 60, 14, 12) // residual
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),  # (256, 60, 7, 6)
            Mixed_5b(),                                                                # (128, 60, 7, 6)
            Mixed_5c(),                                                                # (128, 60, 7, 6)   // residual
            nn.Conv3d(128, 64, kernel_size=1, stride=1, bias=True),                    # (64,  60, 7, 6)
            nn.Dropout3d(p=dropout, inplace=True),                                     # (64,  60, 7, 6)
        )
        '''        
        Mixed_4d(),  # (512, 16, 14, 14)
        Mixed_4e(),  # (528, 16, 14, 14)
        Mixed_4f(),  # (832, 16, 14, 14)
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),  # (832, 8, 7, 7)
        Mixed_5b(),  # (832, 8, 7, 7)
        Mixed_5c(),  # (1024, 8, 7, 7)
        nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),  # (1024, 8, 1, 1)
        nn.Dropout3d(dropout_keep_prob),
        nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),  # (400, 8, 1, 1) '''
        self.fc_out_1 = nn.Linear(64*7*6, 64*6*3)
        self.fc_out_2 = nn.Linear(64*6*3, 64*4*2)
        self.dropout = nn.Dropout(p=0.4)
        self.spatial_squeeze = spatial_squeeze
        self.softmax = nn.Softmax()
        self.device = device
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        #print("initial shape : ", x.shape)
        x = x.permute(0,2,1,3,4).to(self.device)
        logits = self.features(x)

        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        BS = logits.shape[0]

        #print("feature shape : ", logits.shape)
        out = self.relu(self.fc_out_1(self.dropout(logits.permute(0,2,1,3,4).reshape(BS, 60, -1))))
        out = self.fc_out_2(self.dropout(out))

        #print("output shape : ", out.shape)


        return out

    def load_state_dict(self, path):
        target_weights = torch.load(path)
        own_state = self.state_dict()

        for name, param in target_weights.items():

            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    if len(param.size()) == 5 and param.size()[3] in [3, 7]:
                        own_state[name][:, :, 0, :, :] = torch.mean(param, 2)
                    else:
                        own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}.\
                                       whose dimensions in the model are {} and \
                                       whose dimensions in the checkpoint are {}.\
                                       '.format(name, own_state[name].size(), param.size()))
            else:
                print('{} meets error in locating parameters'.format(name))
        missing = set(own_state.keys()) - set(target_weights.keys())

        print('{} keys are not holded in target checkpoints'.format(len(missing)))


if __name__ == '__main__':

    model = S3DG(num_classes=400)

    # Initialize the weights with pretrained I3D net. In detail,
    # please refer to specific reproduced load_state_dict() function
    #if not os.path.exists('modelweights/RGB_imagenet.pkl'):
        #print 'No weights Found! please download first, or comment 382~384th line'
    #model.load_state_dict('modelweights/RGB_imagenet.pkl')
    model = model.cuda()
    data = torch.autograd.Variable(torch.rand(1, 3, 240, 224, 192)).cuda()  # [BS, C, T, H, W]
    basic3d = STConv3d(3, 16, kernel_size=7, stride=2, padding=3).cuda()    # [16, 120, 112, 96]
    '''
    nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))# [16, 120, 56, 48]
    BasicConv3d(64, 64, kernel_size=1, stride=1),                           # [16, 120, 56, 48]
    '''
    print(basic3d(data).shape)
    out = model(data)
    print(out.shape)