# Dynamic Retinex front-end for dynamic MEF

This is a clean-room implementation written for the research idea:

`per-frame exposure decomposition -> reliable cross-exposure correspondence -> real-detail fusion`

## Main structural change

Instead of jointly producing one shared reflectance from two exposure images, every frame is decomposed independently with the **same network parameters**:

- `I_u -> R_u, L_u, G_u`
- `I_m -> R_m, L_m, G_m`
- `I_o -> R_o, L_o, G_o`

The outputs `R_i` preserve each frame's own object position, so the later motion module can estimate:

- `(R_u, R_m) -> V_u2m, C_motion_u`
- `(R_o, R_m) -> V_o2m, C_motion_o`

## Recommended training protocol

### Stage 1: static exposure pretraining

Train `dynamic_retinex_module.py` on spatially aligned static multi-exposure sequences such as SICE patches.

The key loss is cross-exposure reflectance consistency. Because these training images are static and aligned, the model can learn an exposure-robust reflectance representation without mixing motion into the decomposition.

```bash
python train_dynamic_retinex.py \
  --data_root /path/to/SICE_training \
  --save_dir ./exp_dynamic_retinex \
  --epochs 40 \
  --batch_size 8
```

### Stage 2: dynamic MEF

Do **not** apply raw pixelwise `R_i == R_ref` consistency to dynamic frames before alignment.

First extract each frame's `R_i`, then connect the future correspondence module. Once `V_i` and `C_motion_i` exist, a reliability-weighted aligned consistency loss can be added:

`L_align = C_motion_i * |Warp(R_i, V_i->ref) - R_ref|`

Only reliable, non-occluded correspondence regions should contribute.

## Extract reflectance maps

```bash
python extract_reflectance.py \
  --ckpt ./exp_dynamic_retinex/epoch_040.pth \
  --input under.png middle.png over.png \
  --output_dir ./retinex_outputs
```

The output includes `*_R.png`, `*_L.png`, `*_G.png`, and reconstruction images.

## First experiment to run

Before building the full motion module, test the central hypothesis with the same matcher:

1. RGB -> matcher
2. R -> same matcher

Compare forward-backward consistency, warping error on reliable regions, and downstream fusion artifacts. If `R` is not more stable than RGB under extreme exposure, fix the representation before investing in the motion module.
