# Fetal Abdominal Ultrasound Keyframe Detection

This repository targets **keyframe detection** in fetal abdominal ultrasound.  
The pipeline has three modules and should be run in this order:

**Structural Prior Segmentation → Keyframe Detection → Redundancy Suppression**

---

## A. Structural-Prior-Segmentation-Module

**What it does**  
Generates segmentation masks with **nnU-Net** to serve as foreground priors for later feature highlighting.

**Note about nnU-Net**  
- The repo does **not** include nnU-Net training code. Please use the **official nnU-Net v2** data format and commands.  
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
