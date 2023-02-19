import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.pardir)
from model.dcn.modules.modulated_deform_conv import ModulatedDeformConvPack as DCN

def make_model(args):
    return Net()

# 促进特征全方位的融合
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.adaptive = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        s_hw = self.adaptive(x).sigmoid() * 2 # n, c, 1, 1
        x_hw = s_hw * x # n, c, h, w

        x1 = x_hw.permute(0,2,1,3) # n, h, c, w
        s_cw = self.adaptive(x1).sigmoid() * 2  # n, h, 1, 1
        x_cw = (s_cw * x1).permute(0,2,1,3)   # n, h, c, w ===> n, c, h, w

        x2 = x_hw.permute(0,3,2,1) # n, w, h, c
        s_ch = self.adaptive(x2).sigmoid() * 2 # n, w, 1, 1
        x_ch = (s_ch * x2).permute(0,3,2,1)  # n, w, h, c ===> n, c, h, w

        return x_cw + x_hw + x_ch

class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        # spatial attention
        sa_x = self.conv_sa(input_x)  
        # channel attention
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out  = sa_x + ca_x
        return out

# Adaptice Filter Generation 
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels*kernel_size*kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size*self.kernel_size, h, w])

        return filter_x

# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        
    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x = self.afg(input_x)
        unfold_x = self.unfold(input_x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)
        
        return out

# Dynamic residual block
class DRB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.dyconv = DyConv(channels, 3)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.lrelu(self.dyconv(self.conv1(x)))
        # conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out

# residual block
class RB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        out = x + conv3
        return out

class OffsetBlock(nn.Module):
    def __init__(self, in_channels=64, offset_channels=32):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(in_channels, offset_channels, 3, 1, 1) 
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        offset = self.lrelu(self.offset_conv1(x))
        return offset

class DB(nn.Module):
    def __init__(self, in_channels, mid_channels, offset_channels): # [32, 16, 32] [64, 32, 32] [128, 64, 32] [256, 128, 32]
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        if self.in_channels != self.mid_channels:
            self.conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.offset   = OffsetBlock(in_channels, offset_channels)
        self.generate_kernel = nn.Sequential(DCN(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                           extra_offset_mask=True, offset_in_channel=offset_channels),
                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                           SEKG(in_channels, 3),
                           nn.Conv2d(in_channels, mid_channels * 3 **2, 1, 1, 0)
        )
        
        self.lrelu    = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.branch_1 = DepthDC(mid_channels, kernel_size=1)
        self.branch_3 = DepthDC(mid_channels, kernel_size=3)
        self.branch_5 = DepthDC(mid_channels, kernel_size=5)
        self.fusion   = nn.Conv2d(mid_channels*4, in_channels, 1, 1, 0)
        self.attention = Attention()

    def forward(self, x):
        x0 = x
        x_offset = self.offset(x)  
        y = self.generate_kernel([x, x_offset])
        if self.in_channels != self.mid_channels:
            x  = self.conv(x)  #  通道维度进行压缩，低维度通道压缩为一半，高维度保持不变
        x1  = self.branch_1(x, y)
        x3  = self.branch_3(x, y)
        x5  = self.branch_5(x, y)
        out = self.fusion(self.attention(torch.cat([x, x1, x3, x5], dim=1))) + x0
        return out

class DepthDC(nn.Module):
    def __init__(self, in_x_channels, kernel_size):
        super(DepthDC, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3, dilation=kernel_size, padding=kernel_size, stride=1)
        self.fuse = nn.Conv2d(in_x_channels, in_x_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel   = y.reshape([N, xC, 3 ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        out = (unfold_x * kernel).sum(2)
        out = self.lrelu(self.fuse(out))
        return out

class CB(nn.Module):
    def __init__(self, in_channels, mid_channels, offset_channels): 
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.dcn = nn.Sequential(DCN(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                           extra_offset_mask=True, offset_in_channel=offset_channels),
                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                           nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )
        self.offset   = OffsetBlock(in_channels, offset_channels)

        if self.in_channels != self.mid_channels:
            self.conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, 3)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 5, 5)
        self.fusion   = nn.Conv2d(mid_channels*4, in_channels, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attention = Attention()
        
    def forward(self, x):
        offset = self.offset(x)
        x0 = self.dcn([x, offset])

        if self.in_channels != self.mid_channels:
            x0 = self.conv(x0)
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x0))
        x3 = self.lrelu(self.conv3(x0))
        out = self.fusion(self.attention(torch.cat([x0, x1, x2, x3], dim=1))) + x
        return out

class FE(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.fe = nn.Sequential(*[
            RB(in_channels),
            CB(in_channels, mid_channels, offset_channels=32)
        ])

    def forward(self, x):
        out = self.fe(x)
        return out

class DFE(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.fe = nn.Sequential(*[
            DRB(in_channels),
            DB(in_channels, mid_channels, offset_channels=32)
        ])

    def forward(self, x):
        out = self.fe(x)
        return out

class Net(nn.Module):
    def __init__(self, n_colors=3):
        super().__init__()
        n1 = 64
        n2 = 128
        n3 = 256
        n4 = 512
        self.head = nn.Conv2d(n_colors, n1, 3, 1, 1)
        self.fe1  = FE(n1, n1) # 32>>>32
        self.down1 = nn.Conv2d(n1, n2, 3, 2, 1)

        self.fe2 = FE(n2, n2) # 64>>>64
        self.down2 = nn.Conv2d(n2, n3, 3, 2, 1)

        self.fe3 = FE(n3, n3//2) # 128>>>64
        self.down3 = nn.Conv2d(n3, n4, 3, 2, 1)
        
        self.cfe = FE(n4, n4//2) # 256>>>128

        self.up3 = nn.ConvTranspose2d(n4, n3, 6, 2, 2)
        self.dfe3 = DFE(n3, n3//2) # 128>>>64

        self.up2 = nn.ConvTranspose2d(n3, n2, 6, 2, 2)
        self.dfe2 = DFE(n2, n2) # 64>>>64

        self.up1 = nn.ConvTranspose2d(n2, n1, 6, 2, 2)
        self.dfe1 = DFE(n1, n1) # 32>>>32
        
        self.tail = nn.Conv2d(n1, n_colors, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x0 = x
        x = self.head(x)
        conv1 = self.fe1(x)
        pool1 = self.lrelu(self.down1(conv1))

        conv2 = self.fe2(pool1)
        pool2 = self.lrelu(self.down2(conv2))

        conv3 = self.fe3(pool2)
        pool3 = self.lrelu(self.down3(conv3))

        cfe = self.cfe(pool3)

        up3 = self.up3(cfe) + conv3
        dconv3 = self.dfe3(up3)

        up2 = self.up2(dconv3) + conv2
        dconv2 = self.dfe2(up2)

        up1 = self.up1(dconv2) + conv1
        dconv1 = self.dfe1(up1)
        
        out = self.tail(dconv1) + x0

        return out

#==============================================================================#
if __name__ == '__main__':
    model = Net().cuda()
    input = torch.FloatTensor(1, 3, 128, 128).cuda()
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))
    # Params and FLOPs are 28.856675M/46.212299552G

    # from torchsummaryX import summary
    # summary(model, input)
