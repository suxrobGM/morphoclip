<p align="center">
  <h1 align="center">MorphoCLIP</h1>
  <p align="center">
    AI-powered matching of cell microscopy images with text descriptions of biological treatments
    <br />
    <a href="https://morphoclip.suxrobgm.net"><strong>Read the documentation &raquo;</strong></a>
    <br />
    <br />
    <img src="https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white" alt="Python 3.14" />
    <img src="https://img.shields.io/badge/pytorch-2.10+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
    <img src="https://img.shields.io/badge/uv-managed-de5fe9?logo=uv&logoColor=white" alt="uv" />
    <img src="https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=D7FF64" alt="Ruff" />
  </p>
</p>

<br />

## Overview

MorphoCLIP uses AI to connect microscopy images of cells with text descriptions of the treatments applied to them. Given an image of cells treated with a drug or genetic modification, MorphoCLIP can identify what treatment was applied - by learning to match visual patterns in cell images with their textual descriptions.

It combines three key ideas that no prior method unifies:

- **Text supervision** - uses natural language descriptions to guide learning
- **Batch correction** - removes unwanted variation between experimental runs
- **Gene-inclusive training** - learns from both drug treatments and genetic modifications together

We evaluate on the [CPJUMP1 benchmark](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1) (Chandrasekaran et al., 2024), where existing methods detect only 5-25% of expected drug-gene matches.

### How does MorphoCLIP differ from CellCLIP?

[CellCLIP](https://github.com/suinleelab/CellCLIP) (Lu et al., NeurIPS 2025) is the closest prior work - it also uses text-guided learning for cell images. MorphoCLIP builds on CellCLIP's approach but addresses three key limitations:

| | CellCLIP | MorphoCLIP |
|---|---|---|
| **Training data** | Drug treatments only | Drugs + genetic modifications (CRISPR, ORF) |
| **Batch correction** | Applied after training | Built into the training process (CWA) |
| **Vision backbone** | Custom ViT | DINOv3 (stronger pre-trained features) |
| **Text encoder** | General BERT | BioClinical ModernBERT (biomedical-specialized) |

This repo includes a local CellCLIP implementation (`src/cellclip/`) for direct comparison on the same data and evaluation pipeline.

<table>
<tr><td><b>Image model</b></td><td>DINOv3 vision model (300M params) + channel aggregation</td></tr>
<tr><td><b>Text model</b></td><td>BioClinical ModernBERT (150M params) + projection layer</td></tr>
<tr><td><b>Dataset</b></td><td>CPJUMP1 - 51 plates, 3M+ cell images, 303 drugs, 160 genes</td></tr>
</table>

---

## Quick Start

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows: irm https://astral.sh/uv/install.ps1 | iex

# Install all dependencies + PyTorch for your hardware (choose one extra):
uv sync --extra cu128    # CUDA 12.8 (RTX 50-series / Blackwell)
uv sync --extra cu130    # CUDA 13.0
uv sync --extra cpu      # CPU-only (or macOS, which gets MPS wheels)

# Train (tasks run via poe: `uv run poe <task>`)
uv run poe download-features   # download pre-extracted image features
uv run poe precompute-text     # cache text embeddings
uv run poe train               # train MorphoCLIP
```

See the [full documentation](https://morphoclip.suxrobgm.net) for detailed installation, evaluation, inference, and multi-GPU training guides.

---

## Project Structure

```text
src/morphoclip/
  data/           # Dataset, metadata, image loading, splits
  models/         # Image and text encoders, channel aggregation, projection layers
  training/       # Training engine, loss functions, batch correction, config
  utils/          # Text caching, S3 transfer utilities
src/benchmark/    # Benchmark evaluation (metrics, plotting)
src/cellclip/     # CellCLIP baseline (separate from MorphoCLIP)
scripts/          # CLI entry points organized by pipeline stage
configs/
  train/          # MorphoCLIP training configs
  cellclip/       # CellCLIP training configs
  dataset.yml     # Dataset and feature extraction config
tests/            # Mirrors src/ structure
docs/             # Documentation website (Nextra)
```

---

## Authors

- Sukhrobbek Ilyosbekov
- Shubham Gajjar
- Rongfei Jin

---

## Citations

<details>
<summary>BibTeX</summary>

```bibtex
@article{chandrasekaran2024three,
  title={Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations},
  author={Chandrasekaran, Srinivas Niranj and Cimini, Beth A and Goodale, Amy and others},
  journal={Nature Methods},
  volume={21},
  pages={1114--1121},
  year={2024}
}

@inproceedings{lu2025cellclip,
  title={CellCLIP: Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning},
  author={Lu, Mingyu and Weinberger, Ethan and Kim, Chanwoo and Lee, Su-In},
  booktitle={NeurIPS},
  year={2025}
}

@article{huang2025cwamsn,
  title={Efficient Cell Painting Image Representation Learning via Cross-Well Aligned Masked Siamese Network},
  author={Huang, Pin-Jui and Liao, Yu-Hsuan and Kim, SooHeon and Park, NoSeong and Park, JongBae and Shin, DongMyung},
  journal={arXiv:2509.19896},
  year={2025}
}

@article{simeoni2025dinov3,
  title={DINOv3},
  author={Sim{\'e}oni, Oriane and others},
  journal={arXiv:2508.10104},
  year={2025}
}

@article{sounack2025bioclinical,
  title={BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP},
  author={Sounack, Thomas and others},
  journal={arXiv:2506.10896},
  year={2025}
}
```

</details>

## Acknowledgments

- Dataset: [JUMP Cell Painting Consortium](https://jump-cellpainting.broadinstitute.org/)
- Benchmark code: [CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)
