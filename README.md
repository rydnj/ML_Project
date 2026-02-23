# Language-Driven Semantic Segmentation (LSeg) — Reimplementation

A zero-shot semantic segmentation model combining DPT (Dense Prediction Transformer) and CLIP encoders, reimplemented and evaluated as part of graduate coursework at Northeastern University.

## Overview

Traditional semantic segmentation requires expensive pixel-level annotations for every target class. LSeg addresses this by using natural language supervision — allowing the model to segment objects it was never explicitly trained on, guided only by text descriptions.

This project reimplements the [LSeg architecture](https://arxiv.org/abs/2201.03546) and evaluates it under both supervised and zero-shot conditions.

## Results

| Setting | Dataset | Accuracy |
|---------|---------|----------|
| Supervised | COCO Panoptic | 67% |
| Zero-shot | ADE20K | 58% |

The zero-shot result (58% on ADE20K with no pixel-level training labels) validates that language-driven supervision can achieve meaningful segmentation without traditional annotation pipelines.

## Architecture

- **Visual Encoder:** DPT (Dense Prediction Transformer) for pixel-wise dense feature extraction. Initialized with `DPT(head=nn.Identity(), output_feature_dim=C)` where `C` is the desired multimodal embedding dimension.
- **Text Encoder:** CLIP for language-guided class embeddings
- **ViT Adaptation:** Custom `ViTWrapper` enables arbitrary input image sizes by dynamically updating the `image_size` attribute and interpolating learned position embeddings via torchvision's [`interpolate_embeddings`](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L268) utility.
- **Training:** Contrastive alignment between visual features and text embeddings
- **Monitoring:** PyTorch Lightning for telemetry and training metrics
- **Labels:** MSeg universal taxonomy (`universal` column in `mseg-api/mseg/class_remapping_files/MSeg_master.tsv`)

## Tech Stack

Python, PyTorch, PyTorch Lightning, CLIP, DPT, MSeg, Hugging Face Transformers, COCO API

## Setup

**Requirements:** Linux or macOS (mseg dependency constraint)

```bash
pip install -e mseg
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**Data:** Datasets (`COCOPanoptic`, `ADE20K`) must be placed in `data/mseg_dataset/`.

## Usage

**Training** (on HPC cluster):
```bash
sbatch train_script_on_hpc.sh
```
This runs `train.py` with the configured hyperparameters.

**Evaluation:**
Open and run `demo.ipynb` to evaluate the trained model on test images.

## Team

Built collaboratively as a graduate ML course project at Northeastern University (Fall 2024).

## References

- [Language-driven Semantic Segmentation (Li et al., 2022)](https://arxiv.org/abs/2201.03546)
- [Original LSeg Repository](https://github.com/isl-org/lang-seg)
- [MSeg: A Composite Dataset for Multi-domain Semantic Segmentation](https://github.com/mseg-dataset/mseg-api)
