#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience runner for Ours (VTN/FPN):
1) Fix labels in feature NPZs from original GT masks (does NOT touch features).
2) Train the FPN model.
3) Evaluate on the chosen split.
4) Postprocess (if you have your own script).

Paths are controlled via env vars inside each script. Logic is unchanged.
"""

import os

def run_python(script):
    print(f"\n>>> Running: {script}")
    code = os.system(f"python {script}")
    if code != 0:
        raise RuntimeError(f"Script {script} exited with code {code}")

def main():
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())
    os.environ.setdefault("DATA_ROOT", os.path.join(os.environ["PROJECT_ROOT"], "data"))
    # os.environ.setdefault("GT_MASK_DIR", "/abs/path/to/stacked_fetal_abdomen")
    # os.environ.setdefault("INPUT_FEATURE_ROOT", "/abs/path/to/processed_features/features")
    # os.environ.setdefault("OUTPUT_FEATURE_ROOT", os.path.join(os.environ["DATA_ROOT"], "processed_features_gtlabels", "features"))

    # 1) Fix labels using original GT masks
    run_python('Truelabel.py')

    # 2) Train FPN model (uses corrected labels)
    run_python('train.py')

    # 3) Evaluate FPN model
    run_python('evaluate.py')

    # 4) Optional postprocess (your script name preserved)
    run_python('postprocess_diverse.py')

if __name__ == '__main__':
    main()
