#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import json
from typing import Tuple, Optional, List

import numpy as np
import SimpleITK as sitk

# ========= 路径（与前面脚本保持一致）=========
BASE_NNUNET = "/user/work/ad21083/Detection-BSP/Code/SixthVision-Code/nnUnet"
DATA_DIR    = os.path.join(BASE_NNUNET, "data")

SPLIT_DIR   = os.path.join(DATA_DIR, "splits_from_npz")
TMP_IMG_DIR = os.path.join(DATA_DIR, "tmp_for_infer", "images")   # *_0000.mha
TMP_MSK_DIR = os.path.join(DATA_DIR, "tmp_for_infer", "masks")    # 原始 GT，仅备查
INFER_OUT   = os.path.join(DATA_DIR, "infer_results")             # 推理产物目录

FINAL_DIR   = os.path.join(BASE_NNUNET, "new_dataset_pred")       # 目标新数据集
os.makedirs(FINAL_DIR, exist_ok=True)

RANDOM_SEED = 42
MAX_CHECK_PER_SPLIT = 5   # 抽检上限
SAVE_PROBLEM_LIST = True  # 保存缺失/异常名单，便于复查

# ========= 小工具 =========
def read_ids(split: str) -> List[str]:
    p = os.path.join(SPLIT_DIR, f"{split}_list.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing split list: {p}")
    with open(p) as f:
        return [x.strip() for x in f if x.strip()]

def same_header(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize()==b.GetSize() and
            a.GetSpacing()==b.GetSpacing() and
            a.GetOrigin()==b.GetOrigin() and
            a.GetDirection()==b.GetDirection())

def copy_all_metadata(dst: sitk.Image, src: sitk.Image):
    """把 src 的 MetaData 字典复制到 dst（几何信息已由 CopyInformation 处理，这里只复制键值对）"""
    try:
        keys = list(src.GetMetaDataKeys())
    except Exception:
        keys = []
    for k in keys:
        try:
            dst.SetMetaData(k, src.GetMetaData(k))
        except Exception:
            pass  # 不兼容键忽略

def ref_image_for_metadata(case_id: str, split: str) -> Tuple[Optional[str], Optional[str]]:
    """
    返回 (ref_msk_path, ref_img_path)
    优先用 GT mask 作为元信息来源（有些数据会把 label 编码等写进 mask 的 MetaData）
    若没有 GT，则用 image 作为元信息来源
    """
    msk_gt = os.path.join(TMP_MSK_DIR, split, f"{case_id}.mha")
    img    = os.path.join(TMP_IMG_DIR, split, f"{case_id}_0000.mha")
    return (msk_gt if os.path.exists(msk_gt) else None,
            img if os.path.exists(img) else None)

def resample_like(pred: sitk.Image, ref: sitk.Image) -> sitk.Image:
    if same_header(pred, ref):
        return pred
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)   # 标签最近邻
    rs.SetDefaultPixelValue(0)
    return rs.Execute(pred)

def find_prediction_path(split: str, cid: str) -> Optional[str]:
    """
    在 INFER_OUT/{split} 下寻找对应 cid 的预测文件，支持多种后缀：
    1) 精确：cid.nii.gz / cid.mha / cid.nii
    2) 兜底：以 cid 开头、扩展名在允许列表内的第一个匹配
    """
    out_dir = os.path.join(INFER_OUT, split)
    if not os.path.isdir(out_dir):
        return None

    exts = [".nii.gz", ".mha", ".nii"]
    # 精确匹配优先
    for ext in exts:
        p = os.path.join(out_dir, f"{cid}{ext}")
        if os.path.exists(p):
            return p

    # 兜底：cid 开头 + 允许扩展
    cands = []
    for name in os.listdir(out_dir):
        if name.startswith(cid) and (name.endswith(".nii.gz") or name.endswith(".mha") or name.endswith(".nii")):
            cands.append(name)
    cands.sort()
    if cands:
        return os.path.join(out_dir, cands[0])

    return None

def _labels_preview(arr: np.ndarray, max_items: int = 6) -> str:
    """用于日志：返回一个小的标签集合预览"""
    uniq = np.unique(arr)
    if uniq.size > max_items:
        head = ", ".join(map(str, uniq[:max_items]))
        return f"{{{head}, ...}} (total {uniq.size})"
    return "{" + ", ".join(map(str, uniq)) + "}"

def write_mask_with_metadata(pred_path: str, ref_msk_path: Optional[str],
                             ref_img_path: str, out_path: str) -> Tuple[bool, str]:
    """
    将预测结果写成 .mha：
    - 先重采样到 ref_img 几何；
    - 再 CopyInformation(ref_img) 保证 spacing/origin/direction 一致；
    - MetaData 优先从 ref_msk 复制（若存在），否则从 ref_img 复制；
    - 强制 uint8，且不改标签值（不做 2->1 合并）。
    返回 (ok, msg)
    """
    try:
        pred = sitk.ReadImage(pred_path)
        ref_img = sitk.ReadImage(ref_img_path)
        pred = resample_like(pred, ref_img)

        # 预览标签集合，便于排查全 0 或异常值
        preview = _labels_preview(sitk.GetArrayFromImage(pred))

        if pred.GetPixelID() != sitk.sitkUInt8:
            pred = sitk.Cast(pred, sitk.sitkUInt8)

        # 几何信息（size/spacing/origin/direction）
        pred.CopyInformation(ref_img)

        # 复制 MetaData（优先 GT mask）
        try:
            ref_meta_src = sitk.ReadImage(ref_msk_path) if ref_msk_path else ref_img
        except Exception:
            ref_meta_src = ref_img
        copy_all_metadata(pred, ref_meta_src)

        sitk.WriteImage(pred, out_path)
        return True, f"OK; labels={preview}"
    except Exception as e:
        return False, f"ERROR: {e}"

# ========= 主流程 =========
def merge_split(split: str, log: dict):
    ids = read_ids(split)

    img_out = os.path.join(FINAL_DIR, split, "images")
    msk_out = os.path.join(FINAL_DIR, split, "masks_pred")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    # 备份 train 的 GT（仅用于评估/对照）
    gt_out = None
    if split == "train":
        gt_out = os.path.join(FINAL_DIR, "train", "masks_gt")
        os.makedirs(gt_out, exist_ok=True)

    ok, miss_img, miss_pred, fail_write = 0, 0, 0, 0
    missing_images, missing_preds, write_fails = [], [], []

    for cid in ids:
        # 参考几何的 image（*_0000.mha）
        img_src = os.path.join(TMP_IMG_DIR, split, f"{cid}_0000.mha")
        if not os.path.exists(img_src):
            miss_img += 1
            missing_images.append(cid)
            print(f"[WARN] {split}/{cid} missing image (tmp *_0000.mha): {img_src}")
            continue

        # 预测路径
        pred = find_prediction_path(split, cid)
        if pred is None:
            miss_pred += 1
            missing_preds.append(cid)
            print(f"[WARN] {split}/{cid} missing prediction (.nii/.nii.gz/.mha)")
            # 仍然复制 image 出去，保证 images 目录完整
            try:
                img_itk = sitk.ReadImage(img_src)
                sitk.WriteImage(img_itk, os.path.join(img_out, f"{cid}.mha"))
            except Exception as e:
                print(f"[WARN] {split}/{cid} copy image fallback failed: {e}")
            continue

        # 复制 image → {cid}.mha（保持 header + MetaData）
        img_itk = sitk.ReadImage(img_src)
        sitk.WriteImage(img_itk, os.path.join(img_out, f"{cid}.mha"))

        # 元信息来源（优先 GT mask）
        ref_msk, ref_img = ref_image_for_metadata(cid, split)
        if ref_img is None:
            # 极端兜底：如果连 ref_img 都没了就用刚复制的输出图像
            ref_img = os.path.join(img_out, f"{cid}.mha")

        # 写入对齐后的 mask（拷贝 MetaData）
        ok_write, msg = write_mask_with_metadata(pred, ref_msk, ref_img,
                                                 os.path.join(msk_out, f"{cid}.mha"))
        if not ok_write:
            fail_write += 1
            write_fails.append((cid, msg))
            print(f"[ERROR] {split}/{cid} write mask failed: {msg}")
            continue
        else:
            if "labels=" in msg:
                pass  # 如需打印标签集合预览，可在此处 print(msg)

        # 备份 train 的 GT（仅对照用）
        if split == "train" and ref_msk and os.path.exists(ref_msk):
            try:
                sitk.WriteImage(sitk.ReadImage(ref_msk), os.path.join(gt_out, f"{cid}.mha"))
            except Exception as e:
                print(f"[WARN] backup GT failed for {cid}: {e}")

        ok += 1

    log[split] = {
        "total": len(ids),
        "ok": ok,
        "missing_image": miss_img,
        "missing_pred": miss_pred,
        "write_fail": fail_write
    }
    print(f"[{split}] total={len(ids)} ok={ok} miss_img={miss_img} miss_pred={miss_pred} write_fail={fail_write}")

    # 保存问题名单，便于复查/补跑
    if SAVE_PROBLEM_LIST:
        prob_dir = os.path.join(FINAL_DIR, "_problems")
        os.makedirs(prob_dir, exist_ok=True)
        def write_list(name, items):
            with open(os.path.join(prob_dir, f"{split}_{name}.txt"), "w") as f:
                for x in items:
                    f.write(f"{x}\n")
        write_list("missing_images", missing_images)
        write_list("missing_predictions", missing_preds)
        with open(os.path.join(prob_dir, f"{split}_write_fail.txt"), "w") as f:
            for cid, m in write_fails:
                f.write(f"{cid}\t{m}\n")

def quick_checks():
    """
    1) 抽检：每个 split 随机≤5 个（需有 GT），打印 header 是否匹配 & Dice(>0 标签)
    2) 全量体检：images 与 masks_pred 的几何是否一致，打印前 20 条异常
    """
    random.seed(RANDOM_SEED)

    def header_equal(a,b):
        return same_header(a,b)

    # 抽检
    for sp in ["train","val","test"]:
        pred_dir = os.path.join(FINAL_DIR, sp, "masks_pred")
        img_dir  = os.path.join(FINAL_DIR, sp, "images")
        gt_dir   = os.path.join(TMP_MSK_DIR, sp)  # 原始 GT 只用于对照

        if not (os.path.isdir(pred_dir) and os.path.isdir(img_dir)):
            print(f"[Check:{sp}] skip (missing dirs)")
            continue

        gtcases = [f[:-4] for f in os.listdir(gt_dir) if f.endswith(".mha")] if os.path.isdir(gt_dir) else []
        random.shuffle(gtcases)
        sample = gtcases[:MAX_CHECK_PER_SPLIT]

        if not sample:
            print(f"[Check:{sp}] no GT found; skip.")
            continue

        print(f"------ [Check {sp}] sample {len(sample)} cases ------")
        for cid in sample:
            try:
                img = sitk.ReadImage(os.path.join(img_dir,  f"{cid}.mha"))
                msk = sitk.ReadImage(os.path.join(pred_dir, f"{cid}.mha"))
                ok  = header_equal(img, msk)

                gtP = os.path.join(gt_dir, f"{cid}.mha")
                dice = None
                if os.path.exists(gtP):
                    gt = sitk.ReadImage(gtP)
                    # 若几何不同，按图像重采样 GT
                    if not header_equal(gt, img):
                        rs = sitk.ResampleImageFilter()
                        rs.SetReferenceImage(img)
                        rs.SetInterpolator(sitk.sitkNearestNeighbor)
                        rs.SetDefaultPixelValue(0)
                        gt = rs.Execute(gt)
                    g = sitk.GetArrayFromImage(gt)  > 0
                    p = sitk.GetArrayFromImage(msk) > 0
                    inter = (g & p).sum(); s = g.sum() + p.sum()
                    dice = (2.0*inter/s) if s>0 else 1.0
                print(f"[{sp}] {cid}  header_match={ok}  dice={None if dice is None else round(float(dice),4)}")
                if not ok:
                    print(f"      size(img/msk)={img.GetSize()}/{msk.GetSize()} spacing={img.GetSpacing()}/{msk.GetSpacing()}")
            except Exception as e:
                print(f"[Check:{sp}] {cid} ERROR: {e}")

    # 全量体检
    print("\n====== Header audit (full) ======")
    bad = []
    for sp in ["train","val","test"]:
        imgd = os.path.join(FINAL_DIR, sp, "images")
        mskd = os.path.join(FINAL_DIR, sp, "masks_pred")
        if not (os.path.isdir(imgd) and os.path.isdir(mskd)):
            continue
        files = [f for f in os.listdir(imgd) if f.endswith(".mha")]
        for f in files:
            try:
                img = sitk.ReadImage(os.path.join(imgd, f))
                msk = sitk.ReadImage(os.path.join(mskd, f))
                if not same_header(img, msk):
                    bad.append((sp, f,
                                img.GetSize(),   msk.GetSize(),
                                img.GetSpacing(),msk.GetSpacing(),
                                img.GetOrigin(), msk.GetOrigin()))
            except Exception as e:
                bad.append((sp, f, f"ERROR: {e}"))
    print(f"=== mismatches = {len(bad)} ===")
    for rec in bad[:20]:
        print("[Bad]", rec)
    # 想强制失败可在此处：raise RuntimeError(...) / sys.exit(1)

def main():
    summary = {}
    for sp in ["train", "val", "test"]:
        # 推理结果文件夹存在性快速检查
        out_dir = os.path.join(INFER_OUT, sp)
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f"Missing predictions dir: {out_dir}")
        merge_split(sp, summary)

    # 写一份汇总 json
    with open(os.path.join(FINAL_DIR, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[INFO] merge_summary.json written:", os.path.join(FINAL_DIR, "merge_summary.json"))

    # 抽检 + 体检
    quick_checks()

    print("\n✅ 完成！新数据集在：", FINAL_DIR)
    print("   结构：{train,val,test}/{images,masks_pred}（train 另有 masks_gt）")
    print("   推理原始输出：", INFER_OUT)

if __name__ == "__main__":
    main()
