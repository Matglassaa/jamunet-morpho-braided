# # This module stores the code for the CNN U-Net model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2 with 3D Convolution"""

#     def ___init___(self, in_channels, out_channels, mid_channels=None, kernel_size=3, drop_channels=True, p_drop=None):
#         super().___init___()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # 3D convolution layer added here with a smaller temporal kernel size
#         self.conv3d = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size//2, kernel_size//2), bias=False)
        
#         if drop_channels:
#             self.double_conv.add_module('dropout', nn.Dropout2d(p=p_drop))

#     def forward(self, x):
#         x = self.double_conv(x)  # 2D convolution
#         # reshape input for 3D convolution (batch_size, mid_channels, depth = 1, height, width)
#         x = x.unsqueeze(2)
#         x = self.conv3d(x)  # 3D convolution
#         x = x.squeeze(2)  # reshape to original dimension by removing the depth
#         return x

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def ___init___(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None):
#         super().___init___()
#         if pooling == 'max':
#             self.pooling = nn.MaxPool2d(2)
#         elif pooling == 'avg':
#             self.pooling = nn.AvgPool2d(2)
#         self.pool_conv = nn.Sequential(
#             self.pooling,
#             DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
#         )

#     def forward(self, x):
#         return self.pool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def ___init___(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None):
#         super().___init___()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
        
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def ___init___(self, in_channels, out_channels):
#         super(OutConv, self).___init___()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class UNet3D(nn.Module):
#     def ___init___(self, n_channels, n_classes, init_hid_dim=8, kernel_size=3, pooling='max', bilinear=False, drop_channels=False, p_drop=None):
#         super(UNet3D, self).___init___()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.init_hid_dim = init_hid_dim 
#         self.bilinear = bilinear
#         self.kernel_size = kernel_size
#         self.pooling = pooling
#         self.drop_channels = drop_channels
#         self.p_drop = p_drop

#         hid_dims = [init_hid_dim * (2**i) for i in range(5)]
#         self.hid_dims = hid_dims

#         # initial 2D Convolution
#         self.inc = DoubleConv(n_channels, hid_dims[0], kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

#         # downscaling with 2D Convolution followed by pooling
#         self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop)
#         self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop)
#         self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop)

#         # temporal convolution with 3D Convolution
#         self.temporal_conv = nn.Conv3d(hid_dims[3], hid_dims[3], kernel_size=(1, 3, 3), padding=(0, 1, 1))

#         # downscaling with 2D Convolution followed by pooling
#         factor = 2 if bilinear else 1
#         self.down4 = Down(hid_dims[3], hid_dims[4] // factor, kernel_size, pooling, drop_channels, p_drop)

#         # upscaling with 2D Convolution followed by Double Convolution
#         self.up1 = Up(hid_dims[4], hid_dims[3] // factor, kernel_size, bilinear, drop_channels, p_drop)
#         self.up2 = Up(hid_dims[3], hid_dims[2] // factor, kernel_size, bilinear, drop_channels, p_drop)
#         self.up3 = Up(hid_dims[2], hid_dims[1] // factor, kernel_size, bilinear, drop_channels, p_drop)

#         # final 2D Convolution for output
#         self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop)
#         self.outc = OutConv(hid_dims[0], n_classes)
#         # self.relu = nn.ReLU()

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)

#         # temporal 3D Convolution
#         x4_temporal = x4.unsqueeze(2)  # add temporal dimension
#         x4_temporal = self.temporal_conv(x4_temporal)
#         x4 = x4_temporal.squeeze(2)  # remove temporal dimension

#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         # x = self.relu(x)
#         return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv3D(nn.Module):
#     def  ___init___(self, in_ch, out_ch, k=3):
#         super().___init___()
#         p = k // 2
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, k, padding=p, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x): return self.net(x)

# class Down3D(nn.Module):
#     def  ___init___(self, in_ch, out_ch, k=3, pooling='max'):
#         super().___init___()
#         pool = nn.MaxPool3d((1,2,2)) if pooling == 'max' else nn.AvgPool3d((1,2,2))
#         self.net = nn.Sequential(pool, DoubleConv3D(in_ch, out_ch, k))
#     def forward(self, x): return self.net(x)

# class Up3D(nn.Module):
#     def  ___init___(self, in_ch, out_ch, k=3):
#         super().___init___()
#         self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=(1,2,2), stride=(1,2,2))
#         self.conv = DoubleConv3D(in_ch, out_ch, k)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         dD = x2.size(2) - x1.size(2)
#         dH = x2.size(3) - x1.size(3)
#         dW = x2.size(4) - x1.size(4)
#         x1 = F.pad(x1, [dW//2, dW-dW//2, dH//2, dH-dH//2, dD//2, dD-dD//2])
#         return self.conv(torch.cat([x2, x1], dim=1))

# class UNet3D(nn.Module):
#     def  ___init___(self, n_channels, n_classes, init_hid_dim=8, kernel_size=3, pooling='max',
#                  bilinear=False, drop_channels=False, p_drop=None, seq_len=4):
#         super().___init___()
#         base = init_hid_dim
#         self.seq_len = seq_len

#         self.inc   = DoubleConv3D(n_channels, base, kernel_size)
#         self.down1 = Down3D(base,   base*2, kernel_size, pooling)
#         self.down2 = Down3D(base*2, base*4, kernel_size, pooling)
#         self.down3 = Down3D(base*4, base*8, kernel_size, pooling)
#         self.down4 = Down3D(base*8, base*16, kernel_size, pooling)

#         self.up1 = Up3D(base*16, base*8,  kernel_size)
#         self.up2 = Up3D(base*8,  base*4,  kernel_size)
#         self.up3 = Up3D(base*4,  base*2,  kernel_size)
#         self.up4 = Up3D(base*2,  base,    kernel_size)

#         self.collapseD = nn.AdaptiveAvgPool3d((1, None, None))
#         self.outc = nn.Conv3d(base, n_classes, kernel_size=1)

#     def forward(self, x):
#         if x.dim() == 4:
#             x = x.unsqueeze(1)  # (B,1,D,H,W)

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x  = self.up1(x5, x4)
#         x  = self.up2(x,  x3)
#         x  = self.up3(x,  x2)
#         x  = self.up4(x,  x1)

#         x  = self.collapseD(x)
#         x  = self.outc(x).squeeze(2)    # (B, n_classes, H, W)

#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv3D(nn.Module):
#     def ___init___(self, in_ch, out_ch, k=(3,3,3)):
#         super().___init___()
#         p = tuple(ki // 2 for ki in k)
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, k, padding=p, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x): return self.net(x)

# class Down3D(nn.Module):
#     def ___init___(self, in_ch, out_ch, pool_t=False):
#         super().___init___()
#         pool = (2,2,2) if pool_t else (1,2,2)   # optional time downsample
#         self.net = nn.Sequential(nn.MaxPool3d(pool), DoubleConv3D(in_ch, out_ch))
#     def forward(self, x): return self.net(x)

# class Up3D(nn.Module):
#     def ___init___(self, in_ch, out_ch, up_t=False):
#         super().___init___()
#         k = (2,2,2) if up_t else (1,2,2)
#         s = k
#         self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=k, stride=s)
#         self.conv = DoubleConv3D(in_ch, out_ch)
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         dT = x2.size(2) - x1.size(2)
#         dH = x2.size(3) - x1.size(3)
#         dW = x2.size(4) - x1.size(4)
#         x1 = F.pad(x1, [dW//2, dW-dW//2, dH//2, dH-dH//2, dT//2, dT-dT//2])
#         return self.conv(torch.cat([x2, x1], dim=1))

# class UNet3D(nn.Module):
#     """
#     Input : (B,1,T,H,W)
#     Output: (B,2,H,W) logits
#     """
#     def ___init___(self, base=8, n_classes=2, downsample_time_once=True):
#         super().___init___()
#         self.inc   = DoubleConv3D(1, base)
#         self.down1 = Down3D(base,   base*2,  pool_t=False)
#         self.down2 = Down3D(base*2, base*4,  pool_t=False)
#         self.down3 = Down3D(base*4, base*8,  pool_t=downsample_time_once)  # time downsample here
#         self.down4 = Down3D(base*8, base*16, pool_t=False)

#         # extra temporal modeling at bottleneck (small temporal kernels)
#         self.temporal = nn.Sequential(
#             DoubleConv3D(base*16, base*16, k=(3,1,1)),
#             DoubleConv3D(base*16, base*16, k=(3,1,1)),
#         )

#         self.up1 = Up3D(base*16, base*8,  up_t=False)
#         self.up2 = Up3D(base*8,  base*4,  up_t=downsample_time_once)        # restore time if you downsampled it
#         self.up3 = Up3D(base*4,  base*2,  up_t=False)
#         self.up4 = Up3D(base*2,  base,    up_t=False)

#         self.head = nn.Conv3d(base, n_classes, 1)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x5 = self.temporal(x5)

#         x = self.up1(x5, x4)
#         x = self.up2(x,  x3)
#         x = self.up3(x,  x2)
#         x = self.up4(x,  x1)

#         x = self.head(x)      # (B,2,T,H,W)
#         return x.mean(dim=2)  # (B,2,H,W)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down2D(nn.Module):
    def  __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1, bias=False)
        self.conv = DoubleConv2D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1)
        dH = x2.size(2) - x1.size(2)
        dW = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dW//2, dW-dW//2, dH//2, dH-dH//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet3D(nn.Module):
    """
    Input : (B,1,T,H,W)
    Output: (B,n_classes,H,W)
    """
    def __init__(self, base=4, n_classes=2, T=10):
        super().__init__()
        self.stem3d = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm3d(base),
            nn.ReLU(inplace=True),
        )
        self.time_collapse = nn.Conv3d(base, base, kernel_size=(T,1,1), bias=False)

        self.inc   = DoubleConv2D(base,   base)
        self.down1 = Down2D(base,   base*2)
        self.down2 = Down2D(base*2, base*4)
        self.down3 = Down2D(base*4, base*8)

        self.up1 = Up2D(base*8, base*4)
        self.up2 = Up2D(base*4, base*2)
        self.up3 = Up2D(base*2, base)

        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x = self.stem3d(x)              # (B,base,T,H,W)
        x = self.time_collapse(x).squeeze(2)  # (B,base,H,W)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x,  x2)
        x = self.up3(x,  x1)
        return self.outc(x)