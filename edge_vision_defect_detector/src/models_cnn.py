import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class DroneNetSmall(nn.Module):
    # ~90k params
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            conv_bn(1, 16), nn.MaxPool2d(2),
            conv_bn(16, 32), nn.MaxPool2d(2),
            conv_bn(32, 64)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.feat(x)
        return self.head(x).squeeze(1)

class DroneNetMedium(nn.Module):
    # ~300k params
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), nn.MaxPool2d(2),
            conv_bn(64, 128)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.feat(x)
        return self.head(x).squeeze(1)

class DroneNetLarge(nn.Module):
    # ~1.2M params
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(conv_bn(1, 32), conv_bn(32, 32), nn.MaxPool2d(2), SE(32))
        self.stage2 = nn.Sequential(conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2), SE(64))
        self.stage3 = nn.Sequential(conv_bn(64, 128), conv_bn(128, 128), SE(128))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))
    def forward(self, x):
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        return self.head(x).squeeze(1)

# ---- XL variant (depthwise + SE) ----

class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class Block(nn.Module):
    def __init__(self, c, exp=4):
        super().__init__()
        hid = c * exp
        self.pw1 = nn.Conv2d(c, hid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.dw  = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.se  = SE(hid)
        self.pw2 = nn.Conv2d(hid, c, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.act(self.bn1(self.pw1(x)))
        y = self.act(self.bn2(self.dw(y)))
        y = self.se(y)
        y = self.bn3(self.pw2(y))
        return self.act(y + x)

class DroneNetXL(nn.Module):
    # ~3.6M params on 32x32 input
    def __init__(self):
        super().__init__()
        self.stem = conv_bn(1, 32)
        self.s1 = nn.Sequential(conv_bn(32, 64), nn.MaxPool2d(2), Block(64), SE(64))
        self.s2 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Block(128), Block(128), SE(128))
        self.s3 = nn.Sequential(conv_bn(128, 256), Block(256), SE(256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1))
    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return self.head(x).squeeze(1)

def build_model(name: str):
    name = name.lower()
    if name in ["small", "drone_small", "dronenetsmall"]:
        return DroneNetSmall()
    if name in ["med", "medium", "dronenetmedium"]:
        return DroneNetMedium()
    if name in ["large", "big", "dronenetlarge"]:
        return DroneNetLarge()
    if name in ["xl", "xlarge", "dronenetxl"]:
        return DroneNetXL()
    raise ValueError(f"Unknown arch: {name}")
