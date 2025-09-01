import os
import numpy as np
import torch
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ========== 全局配置 ==========
USE_SMALL_SUBSET = False     # 是否仅用小样本
SUBSET_RATIO = 1.0           # 取用比例

def load_mha(file_path):
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image).astype(np.float32)  # (T, H, W)

def get_frame_labels(mask_array):

    labels = np.zeros(mask_array.shape[0], dtype=int)
    for i, frame in enumerate(mask_array):
        if 1 in frame or 2 in frame:
            labels[i] = 1  # 关键帧或次关键帧都合并为1
        else:
            labels[i] = 0  # 背景
    return labels

def enhance_images(images, masks, highlight_factor=0.15):
    masked_images = images.copy()
    masked_images[masks > 0] = (
        masked_images[masks > 0] * (1 - highlight_factor) + highlight_factor * 255
    )
    return masked_images

def process_video(image_path, mask_path):
    images = load_mha(image_path)
    masks = load_mha(mask_path)
    labels = get_frame_labels(masks)  # 只生成合并后的二分类标签
    # enhanced_images = enhance_images(images, masks)
    # return enhanced_images, labels
    return images, labels

def split_dataset(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    train_files, temp_files = train_test_split(file_list, train_size=train_ratio, random_state=seed)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed)
    return train_files, val_files, test_files

# ========== 多进程工作函数 ==========
def process_one(args):
    filename, image_dir, mask_dir, split_dir = args
    img_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)
    save_path = os.path.join(split_dir, filename.replace('.mha', '.npz'))
    try:
        images, labels = process_video(img_path, mask_path)
        np.savez_compressed(save_path, images=images, labels=labels)
        return None
    except Exception as e:
        print(f"[Error] Failed processing {filename}: {e}")
        with open("process_errors.txt", "a") as ferr:
            ferr.write(f"{filename}\n")
        return filename

# ========== 主流程 ==========
if __name__ == "__main__":
    #读取原始数据，如果想研究nnUnet的Mask效果话就读取nnUnet推理后的数据，然后在运行完特征提取后，训练之前先运行Truelabel.py
    image_dir = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/images/stacked_fetal_ultrasound/"
    mask_dir = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen/"
    output_root = "./Datasets/processed_features/"

    os.makedirs(output_root, exist_ok=True)

    all_filenames = [f for f in os.listdir(image_dir)
                     if f.endswith('.mha') and os.path.exists(os.path.join(mask_dir, f))]
    all_filenames = sorted(all_filenames)

    if USE_SMALL_SUBSET:
        subset_len = max(1, int(len(all_filenames) * SUBSET_RATIO))
        all_filenames = all_filenames[:subset_len]
        print(f" Using small subset: {subset_len} samples / Total: {len(all_filenames)}")
    else:
        print(f"Using full dataset: {len(all_filenames)} samples.")

    train_set, val_set, test_set = split_dataset(all_filenames)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    split_map = {'train': train_set, 'val': val_set, 'test': test_set}

    num_workers = min(8, os.cpu_count() or 1)
    for split_name, file_list in split_map.items():
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)

        args_list = [(filename, image_dir, mask_dir, split_dir) for filename in file_list]
        print(f"[INFO] Start processing {split_name} set ({len(args_list)} files, {num_workers} processes)...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_one, args_list), total=len(args_list), desc=f"[{split_name}]"))

    print("All splits processed. Done.")
