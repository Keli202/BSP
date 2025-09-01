import os
import numpy as np
import torch
from tqdm import tqdm
from open_clip import create_model_from_pretrained
from PIL import Image

# ========== 配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json
from open_clip import create_model_and_transforms

model_path = "/user/work/ad21083/Detection-BSP/Model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin"
config_path = "/user/work/ad21083/Detection-BSP/Model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json"

# 加载config
with open(config_path, "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

# 明确指定模型结构
from open_clip.factory import _MODEL_CONFIGS

model_name = "biomedclip_local"
# 注册自定义 config 到 open_clip
if (model_name not in _MODEL_CONFIGS) and (model_cfg is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained=model_path,
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)
model.to(device)
model.eval()


input_root = "Datasets/processed_features"
output_root = "Datasets/processed_features/features"
os.makedirs(output_root, exist_ok=True)

def extract_clip_feature(frame):
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)  # (H, W, 3)
    img = Image.fromarray(frame.astype(np.uint8))
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
    return features.squeeze(0).cpu().numpy()  # (D,)

def process_npz(npz_path):
    data = np.load(npz_path)
    frames = data['images']
    labels = data['labels']
    # 保险：强制所有非0的都设为1
    labels = np.where(labels > 0, 1, 0)
    features = []
    for frame in tqdm(frames, desc=f"Extracting {os.path.basename(npz_path)}"):
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        feat = extract_clip_feature(frame)
        features.append(feat)
    return np.stack(features), labels

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        input_dir = os.path.join(input_root, split)
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)

        file_list = [fname for fname in os.listdir(input_dir) if fname.endswith(".npz")]
        print(f"[INFO] {split} split: {len(file_list)} files")

        for fname in file_list:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            # 跳过已存在的
            if os.path.exists(out_path):
                print(f"[Skip] {out_path} already exists.")
                continue
            try:
                feats, labels = process_npz(in_path)
                np.savez_compressed(out_path, features=feats, labels=labels)
            except Exception as e:
                print(f"[Error] Failed to extract {fname}: {e}")
                with open("extract_feature_errors.txt", "a") as ferr:
                    ferr.write(f"{fname}\n")
