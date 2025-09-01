# Fetal Abdominal Ultrasound Keyframe Detection

This repository targets **keyframe detection** in fetal abdominal ultrasound.  
Run the pipeline in this order:

**Structural Prior Segmentation → Keyframe Detection → Redundancy Suppression**

---

## A. Structural-Prior-Segmentation-Module

**What it does**  
Generates segmentation masks with **nnU-Net** that serve as foreground priors for feature highlighting.

**Note about nnU-Net**  
- This repo does **not** include nnU-Net training code. Please use the **official nnU-Net v2** data format and commands.  
- Training setup: **2D** with `PLANS="nnUNetResEncUNetLPlans"`.  
- For quick reproduction we use **a single fold**. If resources allow, you can train with **multi-fold cross-validation** and ensemble at inference.

**Outputs**  
`nnUnet/new_dataset_pred/{train,val,test}/{images,masks_pred}`

**Minimal usage**
```bash
cd Structural-Prior-Segmentation-Module
python 1-run.py
bash 2-run_nnunet_infer.sh
python 3-runcollect.py
```

---

## B. Keyframe-Detection-Module

### 1) Feature Extraction

**What it does**  
Converts frame sequences into per-frame feature tensors for the Video Transformer (VTN).

#### Basic

**How it works**  
- Extracts features from raw frames by default **without** highlighting.  
- If needed, enable foreground highlighting with nnU-Net masks by following the comments in `process.py`.

**Outputs**  
`Datasets/processed_features/features/{train,val,test}/*.npz`

**Minimal usage**
```bash
cd Keyframe-Detection-Module/FeatureExtraction/Feature-extraction-basic
python process.py
python extract_features.py
# optional diagnostics
python analyze_features.py
python gradcam_visualize.py
```

#### FPN

**How it works**  
- Reads `masks_pred` to highlight foreground regions.  
- Extracts ViT CLS features from layers **3–11**.  
- Per frame: `(9, 768)` and per sequence: `(T, 9, 768)`.

**Outputs**  
`Datasets/processed_features/features/{train,val,test}/*.npz`

**Minimal usage**
```bash
cd Keyframe-Detection-Module/FeatureExtraction/Feature-extraction-FPN
python process.py
python extract_features.py
# optional diagnostics
python analysis.py
python gradcam.py
```

> Paths are defined at the top of each script. Edit them to match your environment.

---

### 2) VTN (Video Transformer)

A transformer that models temporal context over per-frame features.

#### MMsummary (baseline)

**Features used**  
Basic features without highlighting.

**Minimal usage**
```bash
cd Keyframe-Detection-Module/VTN/MMsummary
python train.py
python evaluate.py
```

#### Ours (FPN-VTN)

**Features used**  
FPN features with highlighting and multi-layer ViT descriptors.  
The model includes **four-stage FPN fusion** and **multiple losses**.

**Optional label fix**  
`Truelabel.py` rebuilds frame-level 0/1 labels from GT masks and aligns them to feature length.

**Minimal usage**
```bash
cd Keyframe-Detection-Module/VTN/Ours
# optional but recommended
# python Truelabel.py
# one-click: Truelabel -> train -> evaluate -> postprocess
python run.py
```

---

## C. Redundancy-Suppression-Module

**What it does**  
Loads `*_results.npz` produced by VTN evaluation and performs  
1) candidate selection by probability threshold  
2) redundancy covering by cosine similarity threshold  
3) segment-wise labeling into `0` (background), `1` (keyframe), `2` (sub/edge)  
It also reports WFSS and classification metrics.

**Variants**
- Baseline: `MMsummary-base/postprocess_diverse.py`  
- Ours: `Ours-PRS/postprocess_diverse.py` with peak-preserving smoothing and peak rescue

**Minimal usage**
```bash
# baseline
cd Redundancy-Suppression-Module/MMsummary-base
python postprocess_diverse.py

# ours
cd ../Ours-PRS
python postprocess_diverse.py
```

**Outputs**  
Artifacts under `eval_output/test_*` including frame-wise 3-class predictions, WFSS, PR/F1, confusion matrix and per-case results.  
Thresholds can be changed at the top of each script.

---

## One-click Recommended Flow (Ours)

```bash
# 1) structural prior
cd Structural-Prior-Segmentation-Module
python 1-run.py && bash 2-run_nnunet_infer.sh && python 3-runcollect.py

# 2) FPN features
cd ../Keyframe-Detection-Module/FeatureExtraction/Feature-extraction-FPN
python process.py && python extract_features.py

# 3) train and evaluate
cd ../../VTN/Ours
python run.py

# 4) redundancy suppression
cd ../../../Redundancy-Suppression-Module/Ours-PRS
python postprocess_diverse.py
```

---

## Dependencies

- PyTorch  
- SimpleITK  
- scikit-learn  
- numpy, pandas  
- matplotlib  
- tqdm  
- open_clip  
- pytorch-grad-cam  
- nnU-Net v2

GPU is assumed. If your paths differ, edit the path configuration sections at the top of each script.
