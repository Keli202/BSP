#!/usr/bin/env python3
"""
Purpose
-------
Extract multi-layer ViT CLS features (blocks 3..11) from highlighted frames.
Input  : data/processed_features/{train,val,test}/*.npz  (images, labels)
Output : data/processed_features/features/{split}/*.npz  (features, labels)
Features have shape (T, 9, 768).

Paths
-----
Model/config default to $PROJECT_ROOT/models/...; override with:
  MODEL_PATH, MODEL_CONFIG, PROJECT_ROOT
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import json
from pathlib import Path

from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS

# ---------- paths ----------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

DEFAULT_MODEL_PATH  = REPO_ROOT / "models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_pytorch_model.bin"
DEFAULT_CONFIG_PATH = REPO_ROOT / "models" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_config.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.to(device)
model.eval()

# ViT blocks 3..11 inclusive
BLOCK_IDX_LIST = list(range(3, 12))

def extract_multilayer_feature(frame):
    """Return (9, 768): CLS from ViT blocks 3..11."""
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)
    img = Image.fromarray(frame.astype(np.uint8))
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    outputs = []
    handles = []
    def hook_fn(module, input, output):
        outputs.append(output[:, 0, :].detach().cpu().numpy())  # [CLS]
    for idx in BLOCK_IDX_LIST:
        h = model.visual.trunk.blocks[idx].register_forward_hook(hook_fn)
        handles.append(h)
    with torch.no_grad():
        _ = model.encode_image(img_tensor)
    for h in handles:
        h.remove()
    return np.stack([out[0] for out in outputs], axis=0)  # (9, 768)

def process_npz(npz_path):
    data = np.load(npz_path)
    frames = data['images']
    labels = data['labels']
    labels = np.where(labels > 0, 1, 0)
    features = []
    for frame in frames:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        feat = extract_multilayer_feature(frame)
        features.append(feat)
    return np.stack(features), labels   # (T, 9, 768), (T,)

if __name__ == "__main__":
    input_root  = str(DATA_ROOT / "processed_features")
    output_root = str(DATA_ROOT / "processed_features" / "features")
    os.makedirs(output_root, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_dir = os.path.join(input_root, split)
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)
        file_list = [fname for fname in os.listdir(input_dir) if fname.endswith(".npz")]
        print(f"[INFO] {split} split: {len(file_list)} files")
        for fname in tqdm(file_list, desc=f"Extracting features: {split}"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path):
                print(f"[Skip] {out_path} already exists.")
                continue
            try:
                feats, labels = process_npz(in_path)
                np.savez_compressed(out_path, features=feats, labels=labels)
            except Exception as e:
                print(f"[Error] Failed to extract {fname}: {e}")
                with open("extract_feature_errors.txt", "a") as ferr:
                    ferr.write(f"{fname}: {e}\n")
