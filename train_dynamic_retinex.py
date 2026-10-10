"""Train the clean-room DynamicRetinexDecomposer on aligned static exposure triplets.

Expected directory layout (compatible with a simple SICE-patch style dataset):

root/
  0/
    img_1.png
    img_2.png
    img_3.png
  1/
    img_1.png
    img_2.png
    img_3.png
  ...

All three frames in one folder should depict the same STATIC scene and be spatially aligned.
Do NOT use raw dynamic MEF triplets for the reflectance-consistency loss in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from dynamic_retinex_module import DynamicRetinexDecomposer, LossWeights, static_pretrain_loss


class ExposureTripletDataset(Dataset):
    def __init__(self, root: str, names=("img_1.png", "img_2.png", "img_3.png")):
        self.root = Path(root)
        self.names = names
        self.samples = []
        for p in sorted(self.root.iterdir(), key=lambda q: q.name):
            if p.is_dir() and all((p / n).exists() for n in names):
                self.samples.append(p)
        if not self.samples:
            raise RuntimeError(f"No valid exposure triplets found under {root}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read(path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        folder = self.samples[idx]
        images = torch.stack([self._read(folder / n) for n in self.names], dim=0)
        return images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_dir", default="./exp_dynamic_retinex")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = ExposureTripletDataset(args.data_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = DynamicRetinexDecomposer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    weights = LossWeights()

    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = {}
        for step, images in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            losses = static_pretrain_loss(model, images, weights)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach())

            if step % 20 == 0 or step == len(loader):
                msg = " ".join(f"{k}={meters[k]/step:.5f}" for k in sorted(meters))
                print(f"epoch={epoch:03d} step={step:05d}/{len(loader):05d} {msg}")

        scheduler.step()

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                save_dir / f"epoch_{epoch:03d}.pth",
            )


if __name__ == "__main__":
    main()
