"""UCF101 clip loading for the V-JEPA probe benchmark.

Loads from the HuggingFace mirror `flwrlabs/ucf101` rather than the original
crcv.ucf.edu archive: that host serves an incomplete TLS certificate chain,
which fails verification inside a container (curl error 60), and disabling
verification to fetch a multi-gigabyte archive is not a reasonable trade.

Clips are decoded to a fixed number of uniformly-spaced frames. The split is a
deterministic stratified 80/20 by class, matching how the EuroSAT split is
built here, so every numeric format sees identical data.
"""

import io
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# V-JEPA 2 preprocessing statistics (ImageNet).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

HF_DATASET = "flwrlabs/ucf101"


def _decode_bytes(raw, frames, size):
    """Decode video bytes to (T, C, H, W) float in [0, 1]."""
    import av
    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.video[0]
        all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not all_frames:
        raise ValueError("no frames decoded")
    arr = np.stack(all_frames)                       # (T, H, W, C)
    idx = np.linspace(0, len(arr) - 1, frames).astype(int)
    clip = torch.from_numpy(arr[idx]).permute(0, 3, 1, 2).float() / 255.0
    return torch.nn.functional.interpolate(
        clip, size=(size, size), mode="bilinear", align_corners=False)


class HFClipDataset(Dataset):
    def __init__(self, ds, indices, frames, size, train, video_key, label_key):
        self.ds, self.indices = ds, indices
        self.frames, self.size, self.train = frames, size, train
        self.video_key, self.label_key = video_key, label_key

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec = self.ds[int(self.indices[i])]
        label = int(rec[self.label_key])
        try:
            v = rec[self.video_key]
            raw = v["bytes"] if isinstance(v, dict) else v
            if isinstance(raw, str):                 # a path rather than bytes
                with open(raw, "rb") as fh:
                    raw = fh.read()
            clip = _decode_bytes(raw, self.frames, self.size)
        except Exception:
            # A few UCF101 files are truncated. Skipping them would
            # desynchronise the split, so emit a zero clip instead.
            clip = torch.zeros(self.frames, 3, self.size, self.size)
        if self.train and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])
        return (clip - MEAN) / STD, label


def _keys(example):
    """Identify the video and label columns without hard-coding a schema."""
    vk = next((k for k in ("video", "clip", "mp4", "file", "path")
               if k in example), None)
    lk = next((k for k in ("label", "labels", "class", "target")
               if k in example), None)
    if vk is None or lk is None:
        raise KeyError(f"cannot find video/label columns in {list(example)}")
    return vk, lk


def build_ucf101_loaders(root, frames, size, batch, max_classes, seed,
                         workers=6):
    """`root` is the HF cache directory; the dataset id is fixed."""
    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", root)
    ds = load_dataset(HF_DATASET, split="train")
    vk, lk = _keys(ds[0])

    labels = np.array(ds[lk])
    keep = np.unique(labels)[:max_classes]
    remap = {int(c): i for i, c in enumerate(keep)}

    rng = np.random.RandomState(seed)
    tr_idx, va_idx, tr_y, va_y = [], [], [], []
    for c in keep:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_idx += idx[:cut].tolist()
        va_idx += idx[cut:].tolist()

    # Remap labels to a contiguous range when subsetting classes.
    class _Remapped(HFClipDataset):
        def __getitem__(self, i):
            clip, y = super().__getitem__(i)
            return clip, remap[int(y)]

    common = dict(num_workers=workers, pin_memory=False)
    train = DataLoader(_Remapped(ds, tr_idx, frames, size, True, vk, lk),
                       batch_size=batch, shuffle=False, **common)
    val = DataLoader(_Remapped(ds, va_idx, frames, size, False, vk, lk),
                     batch_size=batch, shuffle=False, **common)
    return train, val, len(keep)
