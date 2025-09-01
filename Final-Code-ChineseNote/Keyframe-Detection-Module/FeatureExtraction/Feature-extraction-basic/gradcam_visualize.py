from skimage.transform import resize
import os
import numpy as np
import torch
from PIL import Image
import json
from open_clip import create_model_and_transforms
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
# ====== 1. 加载模型 ======
model_path = "/user/work/ad21083/Detection-BSP/Model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin"
config_path = "/user/work/ad21083/Detection-BSP/Model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json"

with open(config_path, "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

from open_clip.factory import _MODEL_CONFIGS
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

# ViT patch相关（一般不用动，除非你改了ViT输入size/pool方式）
VIT_SIZE = 224
PATCH_SIZE = 16
FEAT_H = FEAT_W = VIT_SIZE // PATCH_SIZE

target_layers = [model.visual.trunk.blocks[-2]]  # 按你的建议，一般用倒数第二层block
def reshape_transform(tensor, height=FEAT_H, width=FEAT_W):
    # [B, n_patches+1, C] -> [B, C, H, W]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)

def gradcam_on_array(arr, save_path):
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
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
    npz_dir = "Datasets/processed_features/test"
    out_dir = "all_gradcam_outputs"
    os.makedirs(out_dir, exist_ok=True)

    N_KEY, N_BG = 2, 2  # 你可以随时调整
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} npz files in {npz_dir}.")

    for npz_name in npz_files:
        npz_path = os.path.join(npz_dir, npz_name)
        data = np.load(npz_path)
        images, labels = data['images'], data['labels']
        # 所有关键帧和背景帧索引
        key_indices = [i for i, l in enumerate(labels) if l == 1]
        bg_indices  = [i for i, l in enumerate(labels) if l == 0]

        # 随机选N_KEY和N_BG帧
        sampled_key = random.sample(key_indices, min(N_KEY, len(key_indices))) if key_indices else []
        sampled_bg  = random.sample(bg_indices,  min(N_BG,  len(bg_indices)))  if bg_indices  else []

        for idx in sampled_key:
            save_path = os.path.join(out_dir, f"{os.path.splitext(npz_name)[0]}_key_{idx:03d}.jpg")
            gradcam_on_array(images[idx], save_path)
        for idx in sampled_bg:
            save_path = os.path.join(out_dir, f"{os.path.splitext(npz_name)[0]}_bg_{idx:03d}.jpg")
            gradcam_on_array(images[idx], save_path)
        print(f"{npz_name}: {len(sampled_key)} keyframes, {len(sampled_bg)} backgrounds visualized.")

    print(f"All Grad-CAM visualizations saved to {out_dir}")