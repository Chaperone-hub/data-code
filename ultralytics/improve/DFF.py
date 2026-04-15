import torch
import torch.nn as nn


# 自动填充函数（保持不变）
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# 标准卷积模块（保持不变）
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# 修正后的DFF模块：所有层均在__init__预定义
class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 目标输出通道（256，从配置文件传入）

        # 1. 预定义x1通道调整层（128→256，适配scale='n'的输入）
        self.conv_x1 = Conv(128, dim, 1)  # 关键：替换动态创建的层
        # 2. 预定义x2通道调整层（128→256）
        self.conv_up = Conv(128, dim, 1, act=nn.ReLU())

        # 3. 通道注意力层（拼接后通道=256+256=512）
        self.conv_atten = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 4. 特征缩减层（512→256，匹配下一层C3k2的输入）
        self.conv_redu = nn.Conv2d(512, dim, kernel_size=1, bias=False)

        # 5. 空间注意力层（输入通道均为256）
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)  # x1的空间注意力
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)  # x2的空间注意力
        self.nonlin = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 拆分输入的两个特征图（x1：上层输出，x2：backbone特征）
        x1, x2 = x[0], x[1]

        # 关键：使用预定义的层调整通道，无任何动态创建
        x1 = self.conv_x1(x1)  # 128→256（预定义层，已在GPU/半精度）
        x2 = self.conv_up(x2)  # 128→256（预定义层，已在GPU/半精度）

        # 拼接特征图（通道数=256+256=512）
        output = torch.cat([x1, x2], dim=1)

        # 通道注意力计算
        att_channel = self.conv_atten(self.avg_pool(output))
        output = output * att_channel

        # 特征缩减到256通道
        output = self.conv_redu(output)

        # 空间注意力计算
        att_spatial = self.conv1(x1) + self.conv2(x2)
        att_spatial = self.nonlin(att_spatial)
        output = output * att_spatial

        return output


# 测试代码（验证设备和精度匹配）
if __name__ == '__main__':
    # 模拟训练环境：GPU + AMP半精度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 生成半精度输入（匹配AMP训练的输入类型）
    x1 = torch.randn(1, 128, 32, 32).to(device).half()
    x2 = torch.randn(1, 128, 32, 32).to(device).half()

    # 模型移到GPU并转为半精度（与输入一致）
    dff_module = DFF(dim=256).to(device).half()
    output = dff_module((x1, x2))

    # 验证输出类型和形状（应均为半精度，形状正确）
    print(f"Output shape: {output.shape}")  # 预期：torch.Size([1, 256, 32, 32])
    print(f"Output dtype: {output.dtype}")  # 预期：torch.float16（半精度）
    print(f"Weight dtype (conv_x1): {dff_module.conv_x1.conv.weight.dtype}")  # 预期：torch.float16
