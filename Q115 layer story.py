"""
Layer-by-Layer Signal Autopsy
==============================
What actually happens to data as it passes through each layer
of ResNet-20, comparing FP32 inputs vs Q1.15 inputs.

The story told here:
  1. Data enters the network in Q1.15 format
  2. We watch what each layer does to that signal
  3. We compare it to FP32 at every step
  4. We show where and why things break
  5. We show how training accuracy is affected

Run this. Read the output top to bottom. That is the full story.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.manual_seed(42)

Q115_MIN = -1.0
Q115_MAX =  0.999969482421875

def to_q115(x):
    return torch.round(torch.clamp(x, Q115_MIN, Q115_MAX) * 32768) / 32768

class ScaleSymmetric:
    def __call__(self, x): return x * 2.0 - 1.0

# ─────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────

class CIFAR_FP32(Dataset):
    def __init__(self, train=True):
        aug = [transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip()] if train else []
        t = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2470,0.2435,0.2616))
        ])
        self.ds = datasets.CIFAR10('./data', train=train,
                                   download=True, transform=t)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]

class CIFAR_Q115(Dataset):
    def __init__(self, train=True):
        aug = [transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip()] if train else []
        t = transforms.Compose(aug + [transforms.ToTensor(), ScaleSymmetric()])
        self.ds = datasets.CIFAR10('./data', train=train,
                                   download=True, transform=t)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return to_q115(x), y

class GPULoader:
    def __init__(self, ds, bs, shuffle):
        all_x = torch.stack([ds[i][0] for i in range(len(ds))])
        all_y = torch.tensor([ds[i][1] for i in range(len(ds))],
                              dtype=torch.long)
        self.x = all_x.to(DEVICE, dtype=torch.float16)
        self.y = all_y.to(DEVICE)
        self.n, self.bs, self.shuffle = len(all_y), bs, shuffle
    def __iter__(self):
        idx = torch.randperm(self.n, device=DEVICE) if self.shuffle \
              else torch.arange(self.n, device=DEVICE)
        for s in range(0, self.n, self.bs):
            b = idx[s:s+self.bs]
            yield self.x[b].float(), self.y[b]
    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

# ─────────────────────────────────────────────────────────────
# MODEL — ResNet-20 with named intermediate outputs
# ─────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                   padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.layer1 = self._make(16, 16, 3, 1)
        self.layer2 = self._make(16, 32, 3, 2)
        self.layer3 = self._make(32, 64, 3, 2)
        self.fc     = nn.Linear(64, 10)

    def _make(self, i, o, n, s):
        return nn.Sequential(BasicBlock(i,o,s),
                             *[BasicBlock(o,o) for _ in range(n-1)])

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size(3))
        return self.fc(x.view(x.size(0), -1))

    def forward_with_checkpoints(self, x):
        """
        Returns OrderedDict of {layer_name: activation_tensor}
        so we can inspect what each layer produces.
        """
        ckpts = OrderedDict()
        ckpts['0_input']  = x.detach()

        x = F.relu(self.stem(x))
        ckpts['1_stem_out'] = x.detach()

        x = self.layer1(x)
        ckpts['2_layer1_out'] = x.detach()

        x = self.layer2(x)
        ckpts['3_layer2_out'] = x.detach()

        x = self.layer3(x)
        ckpts['4_layer3_out'] = x.detach()

        x = F.avg_pool2d(x, x.size(3))
        x_flat = x.view(x.size(0), -1)
        ckpts['5_pooled'] = x_flat.detach()

        logits = self.fc(x_flat)
        ckpts['6_logits'] = logits.detach()

        return ckpts, logits

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def init_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.0, 0.99)

def clamp_bn(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.clamp_(0.0, Q115_MAX)

scaler = torch.amp.GradScaler('cuda', enabled=True)

def train_model(label, tr_loader, te_loader, epochs=30):
    model = ResNet20().to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    init_bn(model)
    opt = optim.SGD(model.parameters(), lr=0.1,
                    momentum=0.9, weight_decay=1e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, [15, 22], gamma=0.1)

    print(f"\n  Training {label}...")
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                out  = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            clamp_bn(model)
            loss_sum += loss.item() * y.size(0)
            correct  += out.argmax(1).eq(y).sum().item()
            total    += y.size(0)
        sch.step()
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            c = t2 = 0
            with torch.no_grad():
                for x, y in te_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    c  += model(x).argmax(1).eq(y).sum().item()
                    t2 += y.size(0)
            print(f"    Ep {ep:2d} | loss={loss_sum/total:.4f} "
                  f"| train={100*correct/total:.1f}% "
                  f"| test={100*c/t2:.1f}%")
    return model

# ─────────────────────────────────────────────────────────────
# LAYER AUTOPSY
# ─────────────────────────────────────────────────────────────

def stats(t):
    """Return dict of stats for a tensor."""
    t = t.float()
    return {
        'mean'    : t.mean().item(),
        'std'     : t.std().item(),
        'min'     : t.min().item(),
        'max'     : t.max().item(),
        'pct_out' : ((t < Q115_MIN) | (t > Q115_MAX)).float().mean().item() * 100,
        'pct_zero': (t == 0).float().mean().item() * 100,
    }

def bar(pct, width=20, fill='█', empty='░'):
    n = int(pct / 100 * width)
    return fill * n + empty * (width - n)

def print_layer_story(name, fp32_s, q115_s, layer_num, total_layers):
    """Print one layer's worth of the story."""

    pct_out_fp32  = fp32_s['pct_out']
    pct_out_q115  = q115_s['pct_out']
    pct_zero_q115 = q115_s['pct_zero']

    # Determine health status
    if pct_out_q115 > 50:
        status = "CRITICAL — majority of signal outside Q1.15"
        sym    = "✗✗"
    elif pct_out_q115 > 20:
        status = "BAD — large portion of signal lost to clipping"
        sym    = "✗ "
    elif pct_out_q115 > 5:
        status = "WARN — some signal outside Q1.15 range"
        sym    = "! "
    elif pct_out_q115 > 0:
        status = "OK — minor overflow, mostly contained"
        sym    = "~ "
    else:
        status = "CLEAN — all values within Q1.15 range"
        sym    = "✓ "

    print(f"""
  ┌── Layer {layer_num}/{total_layers}: {name}
  │
  │  Signal range comparison:
  │
  │  FP32  min={fp32_s['min']:>8.3f}  max={fp32_s['max']:>7.3f}  std={fp32_s['std']:.3f}
  │  Q1.15 min={q115_s['min']:>8.3f}  max={q115_s['max']:>7.3f}  std={q115_s['std']:.3f}
  │  Q1.15 allowed range: [{Q115_MIN:.1f}, {Q115_MAX:.3f}]
  │
  │  % values outside Q1.15 range:
  │  FP32  [{bar(min(pct_out_fp32,100))}] {pct_out_fp32:5.1f}%
  │  Q1.15 [{bar(min(pct_out_q115,100))}] {pct_out_q115:5.1f}%  ← if these are clipped, info is GONE
  │
  │  % zero activations (dead neurons):
  │  Q1.15 [{bar(min(pct_zero_q115,100))}] {pct_zero_q115:5.1f}%
  │
  │  Status: {sym} {status}
  └──""")

@torch.no_grad()
def run_autopsy(fp32_model, q115_model, fp32_te, q115_te):
    fp32_model.eval()
    q115_model.eval()

    # Get one batch from each test loader
    fp32_iter = iter(fp32_te)
    q115_iter = iter(q115_te)
    x_fp32, _ = next(fp32_iter)
    x_q115, _ = next(q115_iter)
    x_fp32 = x_fp32[:128].to(DEVICE).float()
    x_q115 = x_q115[:128].to(DEVICE).float()

    fp32_ckpts, _ = fp32_model.forward_with_checkpoints(x_fp32)
    q115_ckpts, _ = q115_model.forward_with_checkpoints(x_q115)

    layer_names = {
        '0_input'     : 'Input data (before any processing)',
        '1_stem_out'  : 'Stem: Conv(3→16) + BN + ReLU',
        '2_layer1_out': 'Layer 1: 3× BasicBlock (16 channels, same spatial size)',
        '3_layer2_out': 'Layer 2: 3× BasicBlock (32 channels, 2× downsampled)',
        '4_layer3_out': 'Layer 3: 3× BasicBlock (64 channels, 4× downsampled)',
        '5_pooled'    : 'Global Average Pooling (64-dim vector)',
        '6_logits'    : 'Final Linear → 10 class scores',
    }

    total = len(layer_names)
    print(f"\n{'='*65}")
    print("LAYER-BY-LAYER SIGNAL AUTOPSY")
    print("Showing what happens to data at each layer of ResNet-20")
    print("Left side = FP32 inputs    Right = Q1.15 inputs")
    print(f"{'='*65}")

    for i, (key, name) in enumerate(layer_names.items(), 1):
        fp32_s = stats(fp32_ckpts[key])
        q115_s = stats(q115_ckpts[key])
        print_layer_story(name, fp32_s, q115_s, i, total)

    # Summary: where does the most damage happen?
    print(f"\n{'='*65}")
    print("WHERE THE DAMAGE HAPPENS — Ranked by % out of Q1.15 range")
    print(f"{'='*65}")
    damage = []
    for key, name in layer_names.items():
        q115_s = stats(q115_ckpts[key])
        damage.append((q115_s['pct_out'], name))
    damage.sort(reverse=True)
    for pct, name in damage:
        marker = " ← FIX THIS FIRST" if pct == damage[0][0] else ""
        print(f"  {pct:5.1f}%  {name}{marker}")

# ─────────────────────────────────────────────────────────────
# GRADIENT FLOW AUTOPSY
# ─────────────────────────────────────────────────────────────

def gradient_autopsy(label, tr_loader):
    """
    Run one epoch, record gradient norms per layer group.
    Shows how strongly each layer is learning.
    """
    model = ResNet20().to(DEVICE)
    init_bn(model)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Collect grad norms across one epoch
    grad_norms = {
        'stem'  : [], 'layer1': [], 'layer2': [],
        'layer3': [], 'fc'    : []
    }

    model.train()
    batches = 0
    for x, y in tr_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out  = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # Collect norms per layer group
        for name, param in model.named_parameters():
            if param.grad is None: continue
            norm = param.grad.norm().item()
            if   'stem'   in name: grad_norms['stem'].append(norm)
            elif 'layer1' in name: grad_norms['layer1'].append(norm)
            elif 'layer2' in name: grad_norms['layer2'].append(norm)
            elif 'layer3' in name: grad_norms['layer3'].append(norm)
            elif 'fc'     in name: grad_norms['fc'].append(norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        clamp_bn(model)
        batches += 1
        if batches >= 50: break   # 50 batches is enough to see the pattern

    print(f"\n  Gradient norms per layer group — {label}")
    print(f"  (Higher = layer is learning more actively)")
    print(f"  {'Layer':<12}  {'Mean grad norm':>15}  Visual")
    print(f"  {'-'*12}  {'-'*15}  {'-'*20}")
    for group, norms in grad_norms.items():
        if not norms: continue
        mean_n = np.mean(norms)
        # scale bar to max 0.5 for readability
        scaled = min(mean_n / 0.5, 1.0)
        print(f"  {group:<12}  {mean_n:>15.6f}  {bar(scaled*100)}")

    print()
    print("  If any layer has near-zero gradient norm → it is NOT learning.")
    print("  That layer's weights are frozen. Everything before it is also frozen.")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading data into GPU...")
    fp32_tr = GPULoader(CIFAR_FP32(train=True),  512, shuffle=True)
    fp32_te = GPULoader(CIFAR_FP32(train=False), 512, shuffle=False)
    q115_tr = GPULoader(CIFAR_Q115(train=True),  512, shuffle=True)
    q115_te = GPULoader(CIFAR_Q115(train=False), 512, shuffle=False)
    print("Done.\n")

    # ── PART 1: Train both models ─────────────────────────────
    print("="*65)
    print("PART 1 — TRAINING")
    print("="*65)
    print("Training FP32 model and Q1.15 model for 30 epochs each.")
    print("We need trained models to do the layer autopsy properly.")

    fp32_model = train_model("FP32 inputs", fp32_tr, fp32_te, epochs=30)
    q115_model = train_model("Q1.15 inputs", q115_tr, q115_te, epochs=30)

    # ── PART 2: Gradient flow at epoch 1 (untrained, raw)  ───
    print(f"\n{'='*65}")
    print("PART 2 — GRADIENT FLOW (first 50 batches, untrained weights)")
    print("="*65)
    print("How strongly is each layer learning from Q1.15 vs FP32 inputs?")
    print("Measured at the very start of training, before any learning.")

    gradient_autopsy("FP32 inputs",  fp32_tr)
    gradient_autopsy("Q1.15 inputs", q115_tr)

    # ── PART 3: Full layer-by-layer signal autopsy ────────────
    print(f"\n{'='*65}")
    print("PART 3 — SIGNAL AUTOPSY (trained models, test data)")
    print("="*65)
    print("Passing one batch through each trained model.")
    print("Watching what happens to the signal at every layer.")
    run_autopsy(fp32_model, q115_model, fp32_te, q115_te)

    # ── PART 4: Final accuracy comparison ─────────────────────
    print(f"\n{'='*65}")
    print("PART 4 — FINAL ACCURACY COMPARISON")
    print("="*65)

    def test_acc(model, loader):
        model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                c += model(x).argmax(1).eq(y).sum().item()
                t += y.size(0)
        return 100.0 * c / t

    fp32_acc = test_acc(fp32_model, fp32_te)
    q115_acc = test_acc(q115_model, q115_te)
    gap      = fp32_acc - q115_acc

    print(f"""
  FP32 inputs → FP32 ResNet  :  {fp32_acc:.2f}%
  Q1.15 inputs → FP32 ResNet :  {q115_acc:.2f}%
  Gap                        :  {gap:.2f}%

  What causes this gap?
  → The % of activations outside Q1.15 range at each layer
    (shown in Part 3 above)
  → Those values get clipped if you ever enforce Q1.15 strictly
  → The information in those clipped values is permanently lost
  → The network tries to compensate during training but cannot
    fully recover what was destroyed at input

  What fixes it?
  → Scale input data to fit inside [-1, 1] cleanly      ✓ done
  → Keep BN gamma below 1.0 during training             ✓ done
  → Clip gradients to prevent explosion                 ✓ done
  → The remaining gap is the cost of a fixed-point format
    with no exponent — it cannot represent outlier values at all
    The correct solution for production is QAT with STE
    (training the network to keep its own activations in range)
""")