import modal
import torch
import torch.nn.functional as F
import torch.optim as optim
import csv
import os

# ── Modal Configuration ───────────────────────────────
app = modal.App("sf16-modular-sweep")
volume = modal.Volume.from_name("sf16-cifar-results", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("torch", "torchvision", "pandas")

# Mount the script's specific directory into the container so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
mounts = [modal.Mount.from_local_dir(script_dir, remote_path="/root/project")]

@app.function(image=image, gpu="A10G", timeout=3600 * 24, volumes={"/results": volume}, mounts=mounts)
def train_sweep(depth: int, dataset_name: str):
    import sys
    sys.path.append("/root/project")
    
    from model import ResNetSF, Q115_MAX
    from datasets import get_cifar10, get_cifar100

    DEVICE = torch.device("cuda")
    print(f"[WORKER] Starting SF16 ResNet-{depth} on {dataset_name}...")
    
    if dataset_name.lower() == "cifar10":
        train_loader, val_loader, num_classes = get_cifar10()
    elif dataset_name.lower() == "cifar100":
        train_loader, val_loader, num_classes = get_cifar100()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = ResNetSF(depth=depth, num_classes=num_classes).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=3.1e-5)
    
    best_acc = 0
    log_path = f"/results/{dataset_name}_resnet{depth}_sf16_log.csv"
    ckpt_path = f"/results/{dataset_name}_resnet{depth}_sf16_checkpoint.pth"
    
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"[*] Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        sch.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0)
    else:
        with open(log_path, 'w', newline='') as f: 
            csv.writer(f).writerow(['epoch', 'val_acc'])

    for epoch in range(start_epoch + 1, 201):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            # FIX: Symmetric BN Clamping
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.clamp_(min=-Q115_MAX, max=Q115_MAX)
                        m.bias.clamp_(min=-Q115_MAX, max=Q115_MAX)
        
        sch.step()
        model.eval()
        v_c, v_t = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                v_c += out.argmax(1).eq(y).sum().item()
                v_t += y.size(0)
        
        acc = 100 * v_c / v_t
        if acc > best_acc: best_acc = acc
        
        with open(log_path, 'a', newline='') as f: 
            csv.writer(f).writerow([epoch, acc])
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sch.state_dict(),
            'best_acc': best_acc
        }, ckpt_path)
        
        print(f"ResNet-{depth} {dataset_name} | Ep {epoch:3d}/200 | Val Acc: {acc:5.2f}% | Best: {best_acc:5.2f}%")
        
        if epoch % 5 == 0:
            volume.commit()
            
    volume.commit()
    return f"ResNet-{depth} on {dataset_name} Complete (Best: {best_acc:.2f}%)"

@app.local_entrypoint()
def main():
    import itertools
    depths = [20, 32, 44, 56]
    datasets = ["cifar10", "cifar100"]
    jobs = list(itertools.product(depths, datasets))
    
    print(f"Launching Modular SF16 Sweep! Total Jobs: {len(jobs)}")
    results = list(train_sweep.starmap(jobs))
    print("Sweep Complete:", results)
