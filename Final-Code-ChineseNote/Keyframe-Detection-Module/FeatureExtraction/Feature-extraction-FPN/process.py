#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ========== 路径：直接用 pred 数据集 ==========
#读取nnUnet推理后的数据，注意要在特征提取后训练之前运行Truelabel.py来修正label标签。
BASE_NNUNET  = "../../Structural-Prior-Segmentation-Module/nnUnet-forMask/nnUnet"
DATASET_ROOT = os.path.join(BASE_NNUNET, "new_dataset_pred")  # {train,val,test}/{images,masks_pred}
OUTPUT_ROOT  = "./Datasets/processed_features"                # .npz 输出目录
VIS_ROOT     = os.path.join(BASE_NNUNET, "debug_vis_pred")    # 可视化 PNG 输出

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(VIS_ROOT, exist_ok=True)

# ========== 运行参数 ==========
USE_SMALL_SUBSET = False
SUBSET_RATIO     = 1.0
MAX_WORKERS      = min(8, os.cpu_count() or 1)

# 可视化：从 train 里抽 4 个有前景的 case，每个 2 帧
VIS_ENABLE       = True
VIS_SPLIT        = "train"
VIS_NUM_CASES    = 4
VIS_FRAMES_PER   = 2
HIGHLIGHT_FACTOR = 0.15   # 高亮强度（只在 mask>0 区域把像素往 255 拉近）

# ========== 基本 I/O ==========
def load_mha(path):
    itk = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(itk).astype(np.float32)  # (T,H,W)

def list_cases(images_dir, masks_dir):
    names = [f for f in os.listdir(images_dir) if f.endswith(".mha")]
    names = sorted([f for f in names if os.path.exists(os.path.join(masks_dir, f))])
    return names

# ========== 标签合并（保持你原本语义）==========
def get_frame_labels(mask_array):
    """
    某一帧若出现 label==1 或 label==2 -> 该帧记为 1；否则 0。
    （用 np.any 等价实现，避免 numpy 的 'in' 歧义）
    """
    T = mask_array.shape[0]
    labels = np.zeros(T, dtype=int)
    for i in range(T):
        frame = mask_array[i]
        if np.any(frame == 1) or np.any(frame == 2):
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

# ========== 高亮 ==========
def enhance_images(images, masks, highlight_factor=0.15):
    """
    不加色的提亮：仅在 masks>0 的像素处把亮度向 255 拉近。
    new = old*(1-factor) + 255*factor
    """
    out = images.copy()
    out = np.clip(out, 0, 255)
    sel = masks > 0
    if sel.any():
        out[sel] = out[sel] * (1.0 - highlight_factor) + 255.0 * highlight_factor
    return out.astype(np.float32)

# ========== 你的原流程：返回“高亮后的图 + 合并标签”==========
def process_video(image_path, mask_path):
    images = load_mha(image_path)
    masks  = load_mha(mask_path)
    labels = get_frame_labels(masks)          # 帧级 0/1
    enhanced_images = enhance_images(images, masks, highlight_factor=HIGHLIGHT_FACTOR)
    return enhanced_images, labels

def _worker_npz(args):
    filename, img_dir, msk_dir, out_dir = args
    img_path = os.path.join(img_dir, filename)
    msk_path = os.path.join(msk_dir, filename)
    save_npz = os.path.join(out_dir, filename.replace(".mha", ".npz"))
    try:
        images, labels = process_video(img_path, msk_path)
        np.savez_compressed(save_npz, images=images, labels=labels)
        return None
    except Exception as e:
        with open(os.path.join(out_dir, "process_errors.txt"), "a") as ferr:
            ferr.write(f"{filename}\t{repr(e)}\n")
        return filename

# ========== 可视化（不影响 .npz）==========
def _to_u8_for_show(vol):
    v = vol.astype(np.float32)
    m = np.isfinite(v)
    if not m.any():
        return np.zeros_like(v, dtype=np.uint8)
    lo, hi = np.percentile(v[m], [1, 99])
    if hi <= lo: hi = lo + 1.0
    x = np.clip((v - lo) / (hi - lo), 0, 1)
    return (x * 255.0).round().astype(np.uint8)

def _choose_topk_frames(mask_vol, k=2):
    T = mask_vol.shape[0]
    areas = (mask_vol > 0).reshape(T, -1).sum(1)
    if areas.max() == 0:
        return []
    return list(np.argsort(-areas)[:k])

def _vis_case(save_png, img_vol, pred_vol, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = len(frames)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
    if cols == 1:
        axes = [axes]

    for ax, z in zip(axes, frames):
        base = _to_u8_for_show(img_vol[z])
        highlight = enhance_images(base, (pred_vol[z] > 0).astype(np.uint8),
                                   highlight_factor=HIGHLIGHT_FACTOR).astype(np.uint8)
        ax.imshow(highlight, cmap="gray")
        ax.set_title(f"z={z}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_png, dpi=120)
    plt.close()

# ========== 主流程 ==========
def main():
    # 1) 逐 split 生成 .npz（不重新划分）
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        msk_dir = os.path.join(DATASET_ROOT, split, "masks_pred")
        out_dir = os.path.join(OUTPUT_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)

        files = list_cases(img_dir, msk_dir)
        total = len(files)
        if USE_SMALL_SUBSET:
            keep = max(1, int(total * SUBSET_RATIO))
            files = files[:keep]
            print(f"[{split}] small subset: {keep}/{total}")
        else:
            print(f"[{split}] total: {total}")

        args = [(fn, img_dir, msk_dir, out_dir) for fn in files]
        fails = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for res in tqdm(ex.map(_worker_npz, args), total=len(args), desc=f"[{split}] npz"):
                if res is not None:
                    fails.append(res)
        if fails:
            print(f"[WARN] {split}: {len(fails)} files failed (see process_errors.txt)")
        else:
            print(f"[OK] {split}: all npz done.")

    # 2) 可视化（仅 train，抽有前景的 4 个 case，每个 2 帧）
    if VIS_ENABLE:
        split   = VIS_SPLIT
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        msk_dir = os.path.join(DATASET_ROOT, split, "masks_pred")
        vis_dir = os.path.join(VIS_ROOT, split)
        os.makedirs(vis_dir, exist_ok=True)

        files = list_cases(img_dir, msk_dir)

        # 先找出有前景的 case
        fg_cases = []
        for fn in tqdm(files, desc=f"[{split}] scan FG"):
            pred = load_mha(os.path.join(msk_dir, fn))
            if (pred > 0).any():
                fg_cases.append(fn)
            if len(fg_cases) >= VIS_NUM_CASES:
                break

        for fn in fg_cases:
            cid  = fn[:-4]
            imgs = load_mha(os.path.join(img_dir, fn))
            pred = load_mha(os.path.join(msk_dir, fn))
            frames = _choose_topk_frames(pred, k=VIS_FRAMES_PER)
            if not frames:
                continue
            save_png = os.path.join(vis_dir, f"{cid}.png")
            _vis_case(save_png, imgs, pred, frames)
        print(f"[INFO] PNG saved to: {vis_dir}")

    print("[DONE] npz 构建完成（images=高亮后，labels=合并 0/1）；已从 train 抽样可视化。")

if __name__ == "__main__":
    main()
