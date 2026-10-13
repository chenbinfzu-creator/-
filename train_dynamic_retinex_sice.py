"""Train DynamicRetinexDecomposer directly on the ORIGINAL SICE Part 1 layout.

Expected SICE layout:

Dataset_Part1/
  1/
    1.JPG
    2.JPG
    ...
    7.JPG
  2/
    1.JPG
    ...
  ...
  360/
    ...
  Label/   # or Lable/; ignored automatically
    1.jpg
    ...

For every training sample, this loader:
  1) picks one static scene folder;
  2) randomly chooses K different exposures from that scene (default K=3);
  3) applies the SAME crop/flip to all selected exposures, preserving alignment;
  4) returns a tensor [K, 3, patch_size, patch_size].

The Label/Lable folder is NOT used in this stage because the Retinex front-end is
trained with reconstruction + decomposition constraints rather than fused-image GT.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from dynamic_retinex_module import (
    DynamicRetinexDecomposer,
    LossWeights,
    static_pretrain_loss,
)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _numeric_key(path: Path):
    """Sort 1.JPG, 2.JPG, ... numerically when possible."""
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name.lower())


class SICEOriginalDataset(Dataset):
    """Read the original SICE Part 1 multi-exposure scene folders."""

    def __init__(
        self,
        root: str,
        exposures_per_sample: int = 3,
        patch_size: int = 256,
        samples_per_scene: int = 1,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.exposures_per_sample = exposures_per_sample
        self.patch_size = patch_size
        self.samples_per_scene = samples_per_scene
        self.augment = augment

        if not self.root.exists():
            raise FileNotFoundError(f"SICE root does not exist: {self.root}")
        if exposures_per_sample < 2:
            raise ValueError("exposures_per_sample must be >= 2")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if samples_per_scene <= 0:
            raise ValueError("samples_per_scene must be > 0")

        self.scenes = []
        skipped = []

        # Only numeric directory names are treated as scenes. This automatically
        # ignores Label/, Lable/, README files, etc.
        scene_dirs = [p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()]
        scene_dirs.sort(key=lambda p: int(p.name))

        for scene_dir in scene_dirs:
            images = [
                p for p in scene_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            ]
            images.sort(key=_numeric_key)

            if len(images) < exposures_per_sample:
                skipped.append((scene_dir.name, len(images)))
                continue

            self.scenes.append((scene_dir, images))

        if not self.scenes:
            raise RuntimeError(
                f"No valid numeric SICE scene folders found under {self.root}. "
                f"Expected folders like 1/, 2/, ..., each containing exposure images."
            )

        counts = [len(paths) for _, paths in self.scenes]
        print(
            f"[SICE] found {len(self.scenes)} valid scenes; "
            f"exposures/scene: min={min(counts)}, max={max(counts)}, "
            f"sampling={self.exposures_per_sample} per sample; "
            f"samples_per_scene={self.samples_per_scene}"
        )
        if skipped:
            print(f"[SICE] skipped {len(skipped)} scenes with too few images: {skipped[:5]}")

    def __len__(self):
        # Repeating a scene allows multiple independently sampled triplets/crops
        # from the same multi-exposure sequence in one epoch.
        return len(self.scenes) * self.samples_per_scene

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        # HWC -> CHW
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _resize_min_side(images: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Resize all exposures together only if a dimension is smaller than patch_size."""
        _, _, h, w = images.shape
        if h >= patch_size and w >= patch_size:
            return images

        scale = max(patch_size / h, patch_size / w)
        new_h = max(patch_size, int(round(h * scale)))
        new_w = max(patch_size, int(round(w * scale)))
        return torch.nn.functional.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    def _aligned_random_crop_and_flip(self, images: torch.Tensor) -> torch.Tensor:
        """Apply exactly the same spatial transform to all exposure images."""
        images = self._resize_min_side(images, self.patch_size)
        _, _, h, w = images.shape

        if self.augment:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        images = images[
            :,
            :,
            top : top + self.patch_size,
            left : left + self.patch_size,
        ]

        if self.augment:
            # Same flip for the full exposure group, so pixel alignment is preserved.
            if random.random() < 0.5:
                images = torch.flip(images, dims=[3])  # horizontal
            if random.random() < 0.5:
                images = torch.flip(images, dims=[2])  # vertical

        return images.contiguous()

    def __getitem__(self, idx):
        scene_idx = idx % len(self.scenes)
        scene_dir, exposure_paths = self.scenes[scene_idx]

        # Randomly sample different exposure images from the SAME static scene.
        chosen = random.sample(exposure_paths, self.exposures_per_sample)
        chosen.sort(key=_numeric_key)

        tensors: List[torch.Tensor] = [self._read_rgb(p) for p in chosen]

        # All exposures of a SICE scene should be spatially aligned and have the
        # same resolution. Fail loudly if the downloaded data violates that.
        hw = {(x.shape[-2], x.shape[-1]) for x in tensors}
        if len(hw) != 1:
            shapes = [tuple(x.shape) for x in tensors]
            names = [p.name for p in chosen]
            raise RuntimeError(
                f"Scene {scene_dir.name} has mismatched image sizes for {names}: {shapes}"
            )

        images = torch.stack(tensors, dim=0)  # [T, 3, H, W]
        images = self._aligned_random_crop_and_flip(images)

        return images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Original SICE Dataset_Part1 root")
    p.add_argument("--save_dir", default="./exp_dynamic_retinex_sice")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=1)

    p.add_argument(
        "--exposures_per_sample",
        type=int,
        default=3,
        help="How many different exposures to randomly draw from each scene",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Aligned crop size used for all selected exposures",
    )
    p.add_argument(
        "--samples_per_scene",
        type=int,
        default=4,
        help="How many random exposure/crop samples each scene contributes per epoch",
    )
    p.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable random crop position/flips; use center crop instead",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = SICEOriginalDataset(
        root=args.data_root,
        exposures_per_sample=args.exposures_per_sample,
        patch_size=args.patch_size,
        samples_per_scene=args.samples_per_scene,
        augment=not args.no_augment,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = DynamicRetinexDecomposer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    weights = LossWeights()

    print(
        f"[Train] dataset_samples={len(ds)}, batches/epoch={len(loader)}, "
        f"batch_size={args.batch_size}, patch={args.patch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = {}

        for step, images in enumerate(loader, 1):
            # images: [B, T, 3, H, W]
            images = images.to(device, non_blocking=True)
            losses = static_pretrain_loss(model, images, weights)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach())

            if step % 20 == 0 or step == len(loader):
                msg = " ".join(f"{k}={meters[k] / step:.5f}" for k in sorted(meters))
                print(
                    f"epoch={epoch:03d} step={step:05d}/{len(loader):05d} "
                    f"lr={opt.param_groups[0]['lr']:.2e} grad_norm={float(grad_norm):.3f} {msg}"
                )

        scheduler.step()

        if epoch % args.save_every == 0:
            ckpt_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[Save] {ckpt_path}")


if __name__ == "__main__":
    main()
