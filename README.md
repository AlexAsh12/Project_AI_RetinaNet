# RetinaNet: Object Detection with Focal Loss

Implementation of **RetinaNet** in PyTorch, trained on **PASCAL VOC2007** (20 object classes). Built as Assignment 1 for the CSE4007 Artificial Intelligence course (Hanyang University).

## Overview

RetinaNet addresses the class imbalance problem in single-stage object detectors through **Focal Loss**, which down-weights easy negatives during training. This project implements the full pipeline from data preparation to evaluation:

1. **Data pipeline** — normalization, augmentation, and a custom `VOCDataset` / collate function feeding a PyTorch `DataLoader`
2. **Architecture** — backbone + **Feature Pyramid Network (FPN)** for multi-scale detection
3. **Loss** — Focal Loss for classification, combined with box regression loss

## Repository Contents

| File | Description |
|---|---|
| `assignment1_RetinaNet_FINAL.ipynb` | Main notebook — data pipeline, model, training, evaluation |
| `assignment1_retinanet_FINAL.py` | Script export of the notebook |
| `AI_Assignment_RetinaNet_final_Alexandre_FOLLEAS.pdf` | Written report covering methodology and results |

## Architecture Notes

- Feature Pyramid Network (FPN) built on top of the backbone for detecting objects at multiple scales
- Anchor-based detection with per-level anchor generation
- Focal Loss for classification to counter the extreme foreground/background imbalance typical of dense anchor sets

## Data Pipeline

- Dataset: PASCAL VOC2007
- Custom components: `Normalizer`, `Augmenter`, `VOCDataset`, and a collate function for batching variable-size annotations

## Reproducing / Running

1. Open `assignment1_RetinaNet_FINAL.ipynb` in Google Colab (GPU runtime, T4)
2. Adjust dataset paths for the Colab environment (or local paths if running elsewhere)
3. Ensure `num_anchors` and other config variables are defined before the cells that use them
4. Run the data pipeline, training, and evaluation cells in order

## Practical Notes

- Use `.to(device)` rather than `.cuda()` throughout for CPU/GPU portability
- Load checkpoints with `load_state_dict()` rather than assigning the loaded object directly
- GPU-saved internal buffers may need explicit remapping when reloading on a different device
