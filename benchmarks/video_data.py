"""UCF101 clip loading for the V-JEPA probe benchmark.

Loads from the HuggingFace mirror `flwrlabs/ucf101`. Two notes on that choice:

* The original crcv.ucf.edu archive serves an incomplete TLS certificate chain,
  which fails verification inside a container (curl error 60). Disabling
  verification to fetch a multi-gigabyte archive is not a reasonable trade.
* The mirror stores UCF101 as ~1.79M individual *frames* keyed by
  (video_id, clip_id, frame), not as encoded video. That removes video decoding
  from the pipeline entirely; clips are reassembled here by grouping frames and
  sampling uniformly along each clip's time axis.

The split is a deterministic stratified 80/20 over *source videos*, not clips
and not frames. UCF101 cuts each source video into ~5.25 clips on average
(1,818 videos -> 9,537 clips here), and sibling clips are near-duplicate
segments of the same footage. Splitting at the clip level puts ~4 of a clip's
5 siblings in train while it sits in val, which leaks so badly that an fp32
probe scores 100.00%. Splitting on video_id is what makes the number mean
anything.
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# V-JEPA 2 preprocessing statistics (ImageNet).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

HF_DATASET = "flwrlabs/ucf101"


class ClipDataset(Dataset):
    """Each item is one clip: `frames` uniformly-spaced frames, (T, C, H, W)."""

    def __init__(self, ds, clips, labels, frames, size, train):
        self.ds = ds
        self.clips = clips            # list of np arrays of row indices
        self.labels = labels          # contiguous label per clip
        self.frames, self.size, self.train = frames, size, train

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        rows = self.clips[i]
        pick = np.linspace(0, len(rows) - 1, self.frames).astype(int)
        want = [int(rows[j]) for j in pick]

        imgs = []
        for r in want:
            im = self.ds[r]["image"]
            if im.mode != "RGB":
                im = im.convert("RGB")
            # .copy(): PIL exposes a read-only buffer and torch warns on every frame
            imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
        clip = torch.stack(imgs).float() / 255.0            # (T, C, H, W)
        clip = torch.nn.functional.interpolate(
            clip, size=(self.size, self.size), mode="bilinear",
            align_corners=False)
        if self.train and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])
        return (clip - MEAN) / STD, self.labels[i]


def build_ucf101_loaders(root, frames, size, batch, max_classes, seed,
                         workers=6):
    """`root` is the HF cache dir; the dataset id is fixed."""
    import os

    from datasets import load_dataset

    os.environ.setdefault("HF_HOME", root)
    ds = load_dataset(HF_DATASET, split="train")

    # Columnar reads: pulling three int columns is far cheaper than iterating
    # 1.79M rows through the Python-object path.
    vid = np.asarray(ds["video_id"])
    cid = np.asarray(ds["clip_id"])
    frame_no = np.asarray(ds["frame"])
    labels = np.asarray(ds["label"])

    keep = np.unique(labels)[:max_classes]
    remap = {int(c): i for i, c in enumerate(keep)}

    # Group row indices into clips, ordered along the clip's time axis.
    groups = defaultdict(list)
    sel = np.isin(labels, keep)
    for r in np.nonzero(sel)[0]:
        groups[(vid[r], cid[r])].append(r)

    clip_rows, clip_lab, clip_cls, clip_vid = [], [], [], []
    for (v, _c), rows in groups.items():
        rows = np.asarray(rows)
        rows = rows[np.argsort(frame_no[rows])]
        clip_rows.append(rows)
        clip_lab.append(remap[int(labels[rows[0]])])
        clip_cls.append(int(labels[rows[0]]))
        clip_vid.append(v)

    clip_lab = np.asarray(clip_lab)
    clip_cls = np.asarray(clip_cls)
    clip_vid = np.asarray(clip_vid)

    # Stratified 80/20 over SOURCE VIDEOS: every clip cut from a given video
    # goes to the same side, so near-duplicate siblings cannot straddle it.
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(clip_cls):
        vids_c = np.unique(clip_vid[clip_cls == c])
        rng.shuffle(vids_c)
        cut = max(1, int(0.8 * len(vids_c)))
        tr_v, va_v = set(vids_c[:cut].tolist()), set(vids_c[cut:].tolist())
        for i in np.nonzero(clip_cls == c)[0]:
            (tr if clip_vid[i] in tr_v else va).append(int(i))

    def subset(indices, train):
        return ClipDataset(ds, [clip_rows[i] for i in indices],
                           [int(clip_lab[i]) for i in indices],
                           frames, size, train)

    common = dict(num_workers=workers, pin_memory=False)
    train_ld = DataLoader(subset(tr, True), batch_size=batch, shuffle=False,
                          **common)
    val_ld = DataLoader(subset(va, False), batch_size=batch, shuffle=False,
                        **common)
    print(f"[data] {len(clip_rows)} clips from {len(np.unique(clip_vid))} "
          f"videos over {len(keep)} classes -> train {len(tr)} / val {len(va)} "
          f"(split by source video)", flush=True)
    return train_ld, val_ld, len(keep)
