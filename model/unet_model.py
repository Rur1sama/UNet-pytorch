'''
Author: Rur1sama 1401564074@qq.com
Date: 2025-08-16 11:54:03
LastEditors: Rur1sama 1401564074@qq.com
LastEditTime: 2025-08-24 15:40:38
FilePath: /liwoheng/VS_Project/UNet-pytorch/model/unet_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *

# 定义 U-Net 模型，继承自 PyTorch 的 nn.Module
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels # 输入图像通道数（例如RGB=3，灰度=1）
        self.n_classes = n_classes # 输出类别数（比如二分类=1，多分类>1）
        self.bilinear = bilinear # 上采样方式：双线性插值 or 反卷积
        
        # -------- 编码器（下采样部分） --------
        self.inc = DoubleConv(n_channels, 64)    # 第一层卷积：输入通道 -> 64通道
        self.down1 = Down(64, 128)  # 下采样 1：64 -> 128 通道，特征图尺寸减半
        self.down2 = Down(128, 256) # 下采样 2：128 -> 256 通道
        self.down3 = Down(256, 512) # 下采样 3：256 -> 512 通道
        self.down4 = Down(512, 512) # 下采样 4：512 -> 512 通道（瓶颈层，最深处）
        # -------- 解码器（上采样部分） --------
        # 上采样 1：输入来自 x5(512) 与 x4(512) 拼接 = 1024
        # 输出通道数 256
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # -------- 编码器：逐层下采样 --------
        x1 = self.inc(x) # 第一层特征 (H, W, 64)
        x2 = self.down1(x1) # 第二层特征 (H/2, W/2, 128)
        x3 = self.down2(x2) # 第三层特征 (H/4, W/4, 256)
        x4 = self.down3(x3) # 第四层特征 (H/8, W/8, 512)
        x5 = self.down4(x4) # 最深层特征 (H/16, W/16, 512)

        # -------- 解码器：逐层上采样 + 跳跃连接 --------
        x = self.up1(x5, x4) # 上采样并拼接 x4 特征 (H/8, W/8, 256)
        x = self.up2(x, x3)  # 上采样并拼接 x3 特征 (H/4, W/4, 128)
        x = self.up3(x, x2)  # 上采样并拼接 x2 特征 (H/2, W/2, 64)
        x = self.up4(x, x1)  # 上采样并拼接 x1 特征 (H, W, 64)
        
        # -------- 输出层 --------
        logits = self.outc(x) # 输出 (H, W, n_classes)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)