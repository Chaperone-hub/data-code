import torch
from torch import nn


#
# 论文题目：Multi-Scale and Detail-Enhanced Segment Anything Model for Salient Object Detection
# 论文链接：https://arxiv.org/html/2408.04326v1
# 官方github: https://github.com/BellyBeauty/MDSAM
# 代码改进者：一勺汤

# 定义边缘增强器模块
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        """
        初始化 EdgeEnhancer 模块。

        参数:
        in_dim (int): 输入特征的通道数。
        norm (nn.Module): 归一化层，如 nn.BatchNorm2d。
        act (nn.Module): 激活函数，如 nn.ReLU。
        """
        # 调用父类的构造函数
        super().__init__()
        # 定义输出卷积层，包含卷积、归一化和 Sigmoid 激活函数
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),  # 1x1 卷积层，不使用偏置
            norm(in_dim),  # 归一化层
            nn.Sigmoid()  # Sigmoid 激活函数
        )
        # 定义平均池化层，用于提取特征的局部平均信息
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入的特征图，形状为 (B, C, H, W)。

        返回:
        torch.Tensor: 增强后的特征图，形状为 (B, C, H, W)。
        """
        # 通过平均池化提取特征的局部平均信息
        edge = self.pool(x)
        # 计算边缘信息，即原始特征图与局部平均特征图的差值
        edge = x - edge
        # 通过输出卷积层对边缘信息进行处理
        edge = self.out_conv(edge)
        # 将原始特征图与处理后的边缘信息相加，得到增强后的特征图
        return x + edge

# 定义多尺度边缘增强模块
class MEEM(nn.Module):
    def __init__(self, in_dim):
        """
        初始化 MEEM 模块。

        参数:
        in_dim (int): 输入特征的通道数。
        hidden_dim (int): 隐藏层特征的通道数。
        width (int): 模块的宽度，即中间卷积层和边缘增强器的数量。
        norm (nn.Module): 归一化层，如 nn.BatchNorm2d。
        act (nn.Module): 激活函数，如 nn.ReLU。
        """
        # 调用父类的构造函数
        super().__init__()
        norm = nn.BatchNorm2d
        # 定义激活函数为 ReLU 激活函数
        act = nn.ReLU

        # 保存输入特征的通道数
        self.in_dim = in_dim
        # 保存隐藏层特征的通道数
        hidden_dim = int(in_dim * 0.5)
        # 保存模块的宽度
        self.width = 4
        # 定义输入卷积层，包含卷积、归一化和 Sigmoid 激活函数
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),  # 1x1 卷积层，不使用偏置
            norm(hidden_dim),  # 归一化层
            nn.Sigmoid()  # Sigmoid 激活函数
        )
        # 定义平均池化层，用于提取特征的局部平均信息
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        # 定义中间卷积层列表
        self.mid_conv = nn.ModuleList()
        # 定义边缘增强器列表
        self.edge_enhance = nn.ModuleList()
        # 循环创建中间卷积层和边缘增强器
        for i in range(self.width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),  # 1x1 卷积层，不使用偏置
                norm(hidden_dim),  # 归一化层
                nn.Sigmoid()  # Sigmoid 激活函数
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))
        # 定义输出卷积层，包含卷积、归一化和激活函数
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * self.width, in_dim, 1, bias=False),  # 1x1 卷积层，不使用偏置
            norm(in_dim),  # 归一化层
            act()  # 激活函数
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入的特征图，形状为 (B, C, H, W)。

        返回:
        torch.Tensor: 经过多尺度边缘增强处理后的特征图，形状为 (B, C, H, W)。
        """
        # 通过输入卷积层对输入特征图进行处理
        mid = self.in_conv(x)
        # 初始化输出特征图
        out = mid
        # 循环进行多尺度边缘增强处理
        for i in range(self.width - 1):
            # 通过平均池化提取特征的局部平均信息
            mid = self.pool(mid)
            # 通过中间卷积层对特征图进行处理
            mid = self.mid_conv[i](mid)
            # 通过边缘增强器对特征图进行边缘增强处理，并与之前的输出特征图在通道维度上拼接
            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)
        # 通过输出卷积层对拼接后的特征图进行处理
        out = self.out_conv(out)
        return out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_MEEM(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = MEEM(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_MEEM(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_MEEM(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



if __name__ == "__main__":

    # 创建随机输入特征图，形状为 (B, C, H, W)
    x = torch.randn(2, 32, 17, 31)
    # 创建 MEEM 模块的实例
    meem = MEEM(32)
    # 进行前向传播
    output = meem(x)
    # 打印输出特征图的形状
    print("Output shape:", output.shape)