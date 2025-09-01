#!/usr/bin/env python3
import os, shutil

# ====== 配置 ======
images_dir   = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/images/stacked_fetal_ultrasound"
masks_dir    = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen"
split_dir    = "/user/work/ad21083/Detection-BSP/Code/SixthVision-Code/nnUnet/data/splits_from_npz"   # 自动生成 train_list.txt / val_list.txt / test_list.txt

# 推理输入（保持 .mha，图像重命名为 *_0000.mha；mask 仅备查、不参与推理）
tmp_img_dir  = "/user/work/ad21083/Detection-BSP/Code/SixthVision-Code/nnUnet/data/tmp_for_infer/images"
tmp_mask_dir = "/user/work/ad21083/Detection-BSP/Code/SixthVision-Code/nnUnet/data/tmp_for_infer/masks"

processed_base = "/user/work/ad21083/Detection-BSP/Code/FourthVision-Code/NoMask/Datasets/processed_features"

USE_SYMLINK = True     # 优先用软链接（快、占空间小）；文件系统不支持时会自动回退到复制
SKIP_IF_EXISTS = True  # 目标存在则跳过（提高重跑速度）

os.makedirs(tmp_img_dir,  exist_ok=True)
os.makedirs(tmp_mask_dir, exist_ok=True)

def auto_generate_split_from_existing():
    """
    从已有 processed_features 目录的 train/val/test 生成 split txt
    保证划分与原来一致
    """
    mapping = {
        "train": os.path.join(processed_base, "train"),
        "val":   os.path.join(processed_base, "val"),
        "test":  os.path.join(processed_base, "test")
    }

    os.makedirs(split_dir, exist_ok=True)
    for split, path in mapping.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed features split not found: {path}")
        case_ids = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith(".npz")]
        case_ids.sort()
        with open(os.path.join(split_dir, f"{split}_list.txt"), "w") as f:
            for cid in case_ids:
                f.write(f"{cid}\n")
        print(f"✅ {split} 划分已生成，共 {len(case_ids)} 个样本")

def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if SKIP_IF_EXISTS and os.path.exists(dst):
        return "skip"
    # 清理已有（含坏链）
    if os.path.lexists(dst):
        os.remove(dst)
    if USE_SYMLINK:
        try:
            os.symlink(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"

def prepare_case(case_id, split):
    # 原始路径
    src_img  = os.path.join(images_dir, f"{case_id}.mha")
    src_mask = os.path.join(masks_dir,  f"{case_id}.mha")

    # 目标（注意：图像命名 *_0000.mha，mask 保持原名 .mha）
    dst_img  = os.path.join(tmp_img_dir,  split, f"{case_id}_0000.mha")
    dst_mask = os.path.join(tmp_mask_dir, split, f"{case_id}.mha")

    if not os.path.exists(src_img):
        return False, False, "missing_image"

    img_mode = link_or_copy(src_img, dst_img)
    mask_mode = None
    if os.path.exists(src_mask):
        mask_mode = link_or_copy(src_mask, dst_mask)  # 仅作对照/抽检，不参与推理
        return True, True, f"img:{img_mode},mask:{mask_mode}"
    else:
        return True, False, f"img:{img_mode},mask:None"

def load_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]

def main():
    # 1) 从已有 processed_features 自动生成划分文件
    auto_generate_split_from_existing()

    # 2) 准备 *_0000.mha 图像（供 nnUNet v2 推理）+ 备份 GT（如有）
    for split in ["train", "val", "test"]:
        list_path = os.path.join(split_dir, f"{split}_list.txt")
        cases = load_list(list_path)
        total = len(cases)
        ok, with_mask, miss = 0, 0, 0

        for cid in cases:
            has_img, has_mask, msg = prepare_case(cid, split)
            if not has_img:
                miss += 1
            else:
                ok += 1
                if has_mask:
                    with_mask += 1

        print(f"[{split}] total:{total} ready:{ok} with_GT:{with_mask} missing_img:{miss}")

    print("\n✅ 推理输入已就绪（保持 .mha，图像按 *_0000.mha 命名）：")
    print(f"Images (by split): {tmp_img_dir}")
    print(f"GT masks (by split, optional): {tmp_mask_dir}")
    print("后续 nnUNetv2_predict 直接用以上 images 目录作为 -i 即可。")

if __name__ == "__main__":
    main()
