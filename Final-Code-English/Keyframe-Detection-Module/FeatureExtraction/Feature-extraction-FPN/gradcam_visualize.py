#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM visualizations on sampled frames from NPZs produced by the FPN pipeline.
Reads: data/processed_features/test/*.npz
Saves: ./all_gradcam_outputs/*.jpg

Override model/config via: MODEL_PATH, MODEL_CONFIG, PROJECT_ROOT
"""

from skimage.transform import resize
import os
import numpy as np
import torch
from PIL import Image
import json
from pathlib import Path
from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

# ---------- paths ----------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

DEFAULT_MODEL_PATH  = REPO_ROOT / "models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_pytorch_model.bin"
DEFAULT_CONFIG_PATH = REPO_ROOT / "models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_config.json"
model_path = str(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
config_path = str(os.getenv("MODEL_CONFIG", DEFAULT_CONFIG_PATH))

with open(config_path, "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

model_name = "biomedclip_local"
if (model_name not in _MODEL_CONFIGS) and (model_cfg is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained=model_path,
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

VIT_SIZE = 224
PATCH_SIZE = 16
FEAT_H = FEAT_W = VIT_SIZE // PATCH_SIZE

def reshape_transform(tensor, height=FEAT_H, width=FEAT_W):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)

def gradcam_on_array(arr, block_idx, save_path):
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    target_layers = [model.visual.trunk.blocks[block_idx]]
    with GradCAM(
        model=model.visual,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        grayscale_cam_resized = resize(
            grayscale_cam, arr.shape[:2], order=1, mode='reflect', anti_aliasing=True
        )
        grayscale_cam_resized = (grayscale_cam_resized - grayscale_cam_resized.min()) / (grayscale_cam_resized.max() - grayscale_cam_resized.min() + 1e-8)
        np_img = arr.astype(np.float32) / 255.0
        visualization = show_cam_on_image(np_img, grayscale_cam_resized, use_rgb=True)
        Image.fromarray(visualization).save(save_path)

if __name__ == "__main__":
    npz_dir = str(DATA_ROOT / "processed_features" / "test")
    out_dir = "all_gradcam_outputs"
    os.makedirs(out_dir, exist_ok=True)

    BLOCK_IDX_LIST = [3, 6, 9]  # visualize blocks 4,7,10 conceptually

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} npz files in {npz_dir}.")

    for npz_name in npz_files:
        npz_path = os.path.join(npz_dir, npz_name)
        data = np.load(npz_path)
        images, labels = data['images'], data['labels']
        key_indices = [i for i, l in enumerate(labels) if l == 1]
        bg_indices  = [i for i, l in enumerate(labels) if l == 0]
        sampled_key = [random.choice(key_indices)] if key_indices else []
        sampled_bg  = [random.choice(bg_indices)]  if bg_indices  else []

        for idx in sampled_key:
            for block_idx in BLOCK_IDX_LIST:
                save_path = os.path.join(
                    out_dir,
                    f"{os.path.splitext(npz_name)[0]}_key_{idx:03d}_blk{block_idx+1}.jpg"
                )
                gradcam_on_array(images[idx], block_idx, save_path)
        for idx in sampled_bg:
            for block_idx in BLOCK_IDX_LIST:
                save_path = os.path.join(
                    out_dir,
                    f"{os.path.splitext(npz_name)[0]}_bg_{idx:03d}_blk{block_idx+1}.jpg"
                )
                gradcam_on_array(images[idx], block_idx, save_path)
        print(f"{npz_name}: keyframes({len(sampled_key)}) + backgrounds({len(sampled_bg)}) × {len(BLOCK_IDX_LIST)} layers visualized.")

    print(f"All Grad-CAM visualizations saved to {out_dir}")
