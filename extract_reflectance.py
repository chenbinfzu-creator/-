"""Extract per-frame R/L/G from arbitrary dynamic MEF frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dynamic_retinex_module import DynamicRetinexDecomposer


def read_rgb(path: Path) -> torch.Tensor:
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def save_rgb(x: torch.Tensor, path: Path):
    x = x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_gray(x: torch.Tensor, path: Path):
    x = x.detach().clamp(0, 1).squeeze().cpu().numpy()
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", nargs="+", required=True, help="One or more exposure frames")
    p.add_argument("--output_dir", default="./retinex_outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicRetinexDecomposer().to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for item in args.input:
            path = Path(item)
            image = read_rgb(path).to(device)
            out = model(image)
            stem = path.stem
            save_rgb(out.R, out_dir / f"{stem}_R.png")
            save_gray(out.L, out_dir / f"{stem}_L.png")
            save_rgb(out.G, out_dir / f"{stem}_G.png")
            save_rgb(out.reconstruction, out_dir / f"{stem}_recon.png")
            print(f"saved: {stem}")


if __name__ == "__main__":
    main()
