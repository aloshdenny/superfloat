import torch
import torch.nn as nn
import torch.nn.functional as F

# --- SF16 Hardware Simulation Constants ---
Q115_SCALE = 32768
Q115_MAX   =  32767 / Q115_SCALE
Q115_MIN   = -32767 / Q115_SCALE
HW_EPS     = 1.0 / 16384

def quant_sf16(x):
    x_clamped = torch.clamp(x, Q115_MIN, Q115_MAX)
    x_quant   = torch.round(x_clamped * Q115_SCALE) / Q115_SCALE
    x_quant[x_quant == 0] = 0.0 # Force unique zero
    return x_quant

class QATQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return quant_sf16(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        mask = (x >= Q115_MIN) & (x <= Q115_MAX)
        return grad_output * mask.float()

def sf16(x): 
    return QATQuant.apply(x)

# --- ARCHITECTURE ---
class BasicBlockSF(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch, eps=HW_EPS)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch, eps=HW_EPS)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_ch, eps=HW_EPS)
            )

    def forward(self, x):
        w1 = sf16(self.conv1.weight)
        out = F.conv2d(x.to(torch.float64), w1.to(torch.float64), stride=self.conv1.stride, padding=1, bias=None)
        out = sf16(out)
        
        out = sf16(F.relu(self.bn1(out.to(torch.float32))))
        
        w2 = sf16(self.conv2.weight)
        out2 = F.conv2d(out.to(torch.float64), w2.to(torch.float64), padding=1, bias=None)
        out2 = sf16(out2)
        out2 = sf16(self.bn2(out2.to(torch.float32)))
        
        sc = x
        if len(self.shortcut) > 0:
            w_sc = sf16(self.shortcut[0].weight)
            sc = F.conv2d(x.to(torch.float64), w_sc.to(torch.float64), stride=self.shortcut[0].stride, bias=None)
            sc = sf16(sc)
            sc = sf16(self.shortcut[1](sc.to(torch.float32)))
            
        out = sf16(F.relu(out2 + sc))
        return out

class ResNetSF(nn.Module):
    def __init__(self, depth, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16, eps=HW_EPS)
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlockSF(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks): layers.append(BasicBlockSF(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = sf16(x)
        w1 = sf16(self.conv1.weight)
        x = F.conv2d(x.to(torch.float64), w1.to(torch.float64), padding=1); x = sf16(x)
        
        x = sf16(F.relu(self.bn1(x.to(torch.float32))))
        
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.avg_pool2d(x, 8); x = sf16(x.view(x.size(0), -1))
        
        w_fc = sf16(self.fc.weight); b_fc = sf16(self.fc.bias)
        out = torch.mm(x.to(torch.float64), w_fc.t().to(torch.float64)) + b_fc.to(torch.float64)
        
        return out.to(torch.float32)
