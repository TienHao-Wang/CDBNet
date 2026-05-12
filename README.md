# CDBNet

**CDBNet: A Cross-layer Detail-preserving, Direction-aware, and Boundary-skeleton-guided Network for Chemical Pipeline Extraction from High-Resolution Remote Sensing Images**

📓Note: Before running the code, you need to download the dinov3 code from the dinov3 folder. Download link: https://github.com/facebookresearch/dinov3. This post only includes the official introduction to dinov3 in the dinov3 folder; the download link for the pre-trained weights used in this experiment is provided in weights/cdbnetlink.md.

CDBNet is a deep semantic segmentation framework for extracting **chemical industrial park pipelines** from high-resolution remote sensing images. It is designed for narrow, elongated, low-texture, easily occluded, and topologically complex pipeline structures in dense industrial scenes.
<img width="5688" height="3765" alt="CDFPA" src="https://github.com/user-attachments/assets/6f8e1f4c-57bf-47b0-9cc6-7f675eaf6453" />
<img width="6868" height="3915" alt="framework" src="https://github.com/user-attachments/assets/bf5905f0-4177-4a4f-a963-3cd5b4e51496" />

The model combines a frozen **DINOv3 ViT-L** remote-sensing foundation backbone with three task-specific modules:

- **CDFPA**: Cross-layer Detail-preserving Foundation Pyramid Adapter
- **DAPA**: Direction-Aware Pipeline Aggregation module
- **BSGR**: Boundary-Skeleton Guided Refinement module

CDBNet improves pipeline continuity, boundary localization, and structural integrity in complex chemical industrial park backgrounds.

---

## ⭐Highlights⭐

- Uses a frozen **DINOv3 ViT-L** backbone for strong remote-sensing feature representation.
- Extracts multi-level Transformer features from shallow, middle, and deep layers.
- Preserves fine spatial details through cross-layer feature fusion and image-detail injection.
- Enhances horizontal, vertical, diagonal, curved, and branched pipeline structures using direction-aware aggregation.
- Refines segmentation results with boundary and skeleton cues.
- Evaluated on the self-built **PipelineRS** dataset and public road extraction datasets.
- Achieves strong performance on slender industrial infrastructure extraction.

---

## Overall Architecture

CDBNet follows a feature-extraction, adaptation, decoding, and refinement pipeline:

1. **Frozen DINOv3 ViT-L Backbone**

   The input RGB image is passed through a frozen DINOv3 ViT-L model. Intermediate features are extracted from multiple Transformer layers:

   - shallow layer: local structure and texture
   - middle layer: object morphology
   - deep layer: semantic discrimination

2. **CDFPA: Cross-layer Detail-preserving Foundation Pyramid Adapter**

   CDFPA adapts DINOv3 features for dense segmentation by:

   - projecting multi-layer features into a unified dimension
   - adaptively fusing shallow, middle, and deep features
   - injecting high-resolution image details through gated branches
   - producing a detail-preserving feature pyramid
<img width="5688" height="3765" alt="CDFPA" src="https://github.com/user-attachments/assets/4a94b511-58e5-4c60-b9d3-f796494dee9e" />


3. **DAPA: Direction-Aware Pipeline Aggregation**

   DAPA enhances directional continuity using four complementary branches:

   - horizontal strip convolution
   - vertical strip convolution
   - deformable convolution
   - local 3×3 convolution

   A gated fusion mechanism adaptively weights these branches to recover continuous pipeline structures under different orientations and shapes.
<img width="4082" height="4069" alt="DAPA" src="https://github.com/user-attachments/assets/ea439ef8-878d-4a05-9965-0f719f578fa0" />

4. **BSGR: Boundary-Skeleton Guided Refinement**

   BSGR jointly uses:

   - coarse segmentation masks
   - predicted boundary maps
   - predicted skeleton maps

   The module performs residual refinement to improve boundary accuracy and pipeline topology.
<img width="4027" height="2389" alt="BSGR" src="https://github.com/user-attachments/assets/785897d0-95cf-46f7-8a18-ad9ad1ad034b" />

---

## Main Contributions

- A new remote sensing pipeline extraction dataset, **PipelineRS**, was constructed from GF-2 and GF-7 satellite imagery covering four representative chemical industrial parks in China.
- A detail-preserving adaptation strategy, **CDFPA**, was proposed to bridge frozen DINOv3 features and dense segmentation requirements.
- A direction-aware module, **DAPA**, was designed to enhance elongated pipeline responses in multiple geometric directions.
- A boundary-skeleton-guided refinement module, **BSGR**, was introduced to improve structural integrity and boundary localization.
- CDBNet was evaluated on chemical pipeline extraction and road extraction datasets, demonstrating strong accuracy, robustness, and transferability.
<img width="3215" height="2233" alt="Fig1" src="https://github.com/user-attachments/assets/72bcf6c2-e0e1-457c-a07b-de215cbfa9a5" />
---

## Performance on PipelineRS

| Model | Precision | Recall | F1-Score | FIoU | MIoU | ALPS |
|---|---:|---:|---:|---:|---:|---:|
| CDBNet | 0.9499 | 0.9365 | 0.9432 | 0.8925 | 0.9460 | 0.9029 |

---

## Ablation Study

| CDFPA | DAPA | BSGR | Precision | Recall | F1-Score | FIoU | MIoU | ALPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ✗ | ✗ | ✗ | 0.8866 | 0.8678 | 0.8771 | 0.7812 | 0.8901 | 0.7920 |
| ✓ | ✗ | ✗ | 0.9014 | 0.8948 | 0.8981 | 0.8151 | 0.9071 | 0.8446 |
| ✗ | ✓ | ✗ | 0.9203 | 0.9046 | 0.9124 | 0.8389 | 0.9191 | 0.8588 |
| ✗ | ✗ | ✓ | 0.9151 | 0.8949 | 0.9048 | 0.8262 | 0.9127 | 0.8146 |
| ✓ | ✓ | ✗ | 0.9126 | 0.9212 | 0.9169 | 0.8466 | 0.9229 | 0.8757 |
| ✓ | ✗ | ✓ | 0.9330 | 0.9296 | 0.9313 | 0.8714 | 0.9354 | 0.8872 |
| ✗ | ✓ | ✓ | 0.9203 | 0.9155 | 0.9179 | 0.8482 | 0.9237 | 0.8587 |
| ✓ | ✓ | ✓ | 0.9499 | 0.9365 | 0.9432 | 0.8925 | 0.9460 | 0.9029 |

---

## Dataset

### PipelineRS

PipelineRS is a high-resolution remote sensing dataset for chemical industrial park pipeline extraction.

The dataset is built from RGB optical imagery captured by Chinese high-resolution Earth observation satellites:

- GF-2
- GF-7

It covers four representative chemical industrial parks:

| Park | Province | Geographic Zone | Scene Characteristics |
|---|---|---|---|
| Meishan | Sichuan | Southwest inland basin | light fog, aerosols, reduced contrast |
| Ningbo | Zhejiang | Southeast coastal hills | moist air, salt spray, local overexposure |
| Weifang | Shandong | North China Plain | complex industrial background |
| Yantai | Shandong | North China coastal plain | densely intertwined pipelines |

### Dataset Statistics

- Total image-label pairs: **18,832**
- Image size: **512 × 512**
- Image type: RGB
- Label type: binary mask
- Foreground: pipeline pixels, value `255`
- Background: non-pipeline pixels, value `0`
- Dataset split: **train : val : test = 7 : 1 : 2**
- In accordance with the research team's requirements, only a portion of the dataset has been made open source.

### Expected Dataset Structure

Please organize the dataset as follows:

```text
datasets/
└── PipelineRS/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Each image should have a corresponding mask with the same filename.

Example:

```text
datasets/PipelineRS/train/images/NingboGF7cj_tile_03712.tif
datasets/PipelineRS/train/labels/NingboGF7cj_tile_03712.tif
```

---

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended core dependencies:

```text
torch
torchvision
timm
opencv-python
numpy
scipy
scikit-image
albumentations
tqdm
matplotlib
einops
```

---

## Pretrained Backbone

CDBNet uses a frozen **DINOv3 ViT-L** backbone pretrained on remote sensing imagery.


Please download the DINOv3 ViT-L checkpoint and place it under:

```text
pretrained/
└── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```
**The pre-trained weights for dinov3 can be downloaded from official sources.**
Then update the checkpoint path in the configuration file:

```yaml
MODEL:
  BACKBONE: dinov3_vitl
  BACKBONE_WEIGHTS: dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
  FREEZE_BACKBONE: true
```

---

## Training

Train CDBNet on PipelineRS:

```bash
python train.py \
  --config configs/cdbnet_pipelinenrs.yaml \
  --data-root datasets/PipelineRS \
  --save_dir checkpoints_rs_cdbnet
```

Default training settings from the paper:

| Hyperparameter | Value |
|---|---:|
| Batch size | 16 |
| Training iterations | 100 |
| Initial learning rate | 0.0005 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Learning rate schedule | Cosine Annealing |
| GPU | NVIDIA GeForce RTX 3090 24GB |

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python tools/evaluate.py 
```

The evaluation reports:

- Precision
- Recall
- F1-Score
- Foreground IoU
- Mean IoU
- ALPS
- Connectivity
- Completeness

---

## Custom Dataset Preparation

To train CDBNet on your own binary segmentation dataset, prepare images and masks using the following structure:

```text
datasets/
└── YourDataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Mask requirements:

- background: `0`
- foreground target: `255`
- image and mask filenames should match
- recommended crop size: `512 × 512`

Then modify the dataset path in the config:

```yaml
DATASET:
  NAME: YourDataset
  ROOT: datasets/YourDataset
  IMAGE_SIZE: 512
  NUM_CLASSES: 1
```

---

## Metrics

### Precision

Measures the proportion of correctly predicted pipeline pixels among all predicted pipeline pixels.

### Recall

Measures the proportion of ground-truth pipeline pixels correctly detected by the model.

### F1-Score

The harmonic mean of Precision and Recall.

### FIoU

Foreground Intersection over Union, computed only on the pipeline class.

### MIoU

Mean Intersection over Union, computed over foreground and background classes.

### ALPS

Average Boundary Localization Accuracy.  
This metric evaluates the alignment between predicted boundaries and ground-truth boundaries. A higher ALPS indicates better boundary preservation.

---

## Robustness

CDBNet was evaluated under multiple image degradation conditions, including:

- motion blur
- Gaussian blur
- haze
- shadow occlusion
- brightness variation
- JPEG compression
- sensor noise
- local occlusion
- color shift

The model remains relatively stable under brightness variation, local occlusion, Gaussian blur, haze, and shadow conditions, but performance decreases under severe motion blur, JPEG compression, and sensor noise.

---

## Cross-Dataset Generalization

CDBNet was also evaluated on public road extraction datasets:

- Massachusetts Road Dataset
- DeepGlobe Road Dataset

Since roads and chemical pipelines are both elongated man-made structures with strong topological continuity, these experiments demonstrate that CDBNet can transfer to other linear-object segmentation tasks.

### DeepGlobe Road Dataset

| Model | Precision | Recall | F1-Score | FIoU | MIoU | ALPS |
|---|---:|---:|---:|---:|---:|---:|
| D-LinkNet | 0.7895 | 0.7779 | 0.7837 | 0.6443 | 0.8130 | 0.8018 |
| MADSNet | 0.7901 | 0.7897 | 0.7899 | 0.6528 | 0.8175 | 0.8045 |
| CDBNet | 0.8050 | 0.8080 | 0.8065 | 0.6757 | 0.8296 | 0.8317 |

---


## Citation

If you find this project useful, please cite:

```bibtex
@article{wang2026cdbnet,
  title={CDBNet: A Cross-layer Detail-preserving, Direction-aware, and Boundary-skeleton-guided Network for Chemical Pipeline Extraction from High-Resolution Remote Sensing Images},
  author={Wang, Tianhao and Wang, Shixin and Wang, Futao and Li, Suju and Wang, Zhenqing and Wang, Litao and Gu, Xingguang and Nie, Ziqi and Wang, Zhaowei and Xiong, Chengyue and Tao, Haojie and Zhu, Jinfeng and Liu, Wenliang},
  url={https://github.com/TienHao-Wang/CDBNet},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

Please note that the dataset and pretrained backbone checkpoints may be subject to separate licenses or access restrictions.

---

## Acknowledgements😄

This work was supported by:

- National Key Research and Development Program of China
- Yunnan Sci-Tech Talent and Platform Plan Project

We thank the contributors and researchers involved in remote sensing image interpretation, visual foundation models, road extraction, and slender-object segmentation.

---

## Contact

For questions about the paper, code, or dataset, please contact:

```text
Author: Tianhao Wang
Email: wangtianhao24@mails.ucas.ac.cn
```
