---
title: Plant Leaf Super Resolution
emoji: 🌿
colorFrom: green
colorTo: leaf
sdk: docker
app_port: 7860
pinned: false
---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Gradio-4.0+-f97316?logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/🤗_Transformers-4.36+-fbbf24" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/HF_Spaces-deployed-ff9d00?logo=huggingface&logoColor=white" alt="HF Spaces"/>
</p>

# 🌿 Plant Leaf Super-Resolution

> Upscale low-resolution plant leaf images to high resolution using deep-learning-based super-resolution.

A **portfolio demo** by [Yash Chavan](https://github.com/cyash24f3) showcasing the [Plant Leaf Super-Resolution](https://github.com/cyash24f3/plant-super-resolution) Kaggle competition project. The original model uses an **RRDB + CBAM GAN** architecture (similar to Real-ESRGAN) with 16 Residual-in-Residual Dense Blocks, Channel & Spatial Attention, and a combined Perceptual + Adversarial + L1 loss. This demo substitutes a pretrained **Swin2SR** transformer for easy deployment without custom weights.

---

## ✨ Features

| Feature | Details |
|---|---|
| **2× Super-Resolution** | Lightweight Swin2SR model for high-quality upscaling |
| **REST API** | FastAPI endpoints for health checks and programmatic upscaling |
| **Polished UI** | Gradio Blocks with dark botanical theme, side-by-side comparison |
| **Containerised** | Multi-stage Dockerfile, ready for Hugging Face Spaces |
| **Zero Config** | Model weights auto-download from HuggingFace Hub |

---

## 📸 Screenshots

<!-- Add screenshots here -->
<!-- ![Demo Screenshot](assets/screenshot.png) -->

---

## 🏗️ Architecture

```
Original competition model (Yash Chavan):
┌─────────────────────────────────────────────────┐
│  Generator: 16× RRDB + CBAM (Channel+Spatial)  │
│  Discriminator: VGG-style PatchGAN              │
│  Losses: VGG-19 Perceptual + Adversarial + L1   │
└─────────────────────────────────────────────────┘

This demo uses:
┌─────────────────────────────────────────────────┐
│  Swin2SR-Lightweight (2×)                       │
│  caidas/swin2SR-lightweight-x2-64               │
│  ~1M parameters, Swin Transformer V2 backbone   │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Backend** — FastAPI + Uvicorn
- **Frontend** — Gradio 4 Blocks (dark emerald theme)
- **Model** — Swin2SR via 🤗 Transformers
- **Deep Learning** — PyTorch 2
- **Image Processing** — Pillow, NumPy, OpenCV
- **Container** — Docker (multi-stage, Python 3.11-slim)
- **Deployment** — Hugging Face Spaces (Docker SDK)

---

## 🚀 Quick Start

### Local (Python)

```bash
# Clone
git clone https://github.com/cyash24f3/plant-super-resolution.git
cd plant-super-resolution

# Install
pip install -r requirements.txt

# Run
python -m app.main
```

Open [http://localhost:7860](http://localhost:7860).

### Docker

```bash
docker build -t plant-sr .
docker run -p 7860:7860 plant-sr
```

---

## 📡 API Reference

### Health Check

```
GET /api/health
```

```json
{
  "status": "ok",
  "model": "Swin2SR",
  "model_name": "caidas/swin2SR-lightweight-x2-64",
  "device": "cpu",
  "upscale_factor": 2
}
```

### Upscale Image

```
POST /api/upscale
Content-Type: multipart/form-data
Body: file=<image>
```

Returns the super-resolved image as `image/png` with metadata headers:

| Header | Example |
|---|---|
| `X-Input-Size` | `256x256` |
| `X-Output-Size` | `512x512` |
| `X-Upscale-Factor` | `2×` |

Interactive docs: [/api/docs](http://localhost:7860/api/docs)

---

## 🐳 Hugging Face Spaces Deployment

This project is configured for **HF Spaces with Docker SDK**.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repo to the Space
4. The container auto-builds and starts on port `7860`

**Live demo:** [huggingface.co/spaces/cyash1204/plant-super-resolution](https://huggingface.co/spaces/cyash1204/plant-super-resolution)

---

## 📁 Project Structure

```
plant-super-resolution/
├── app/
│   ├── __init__.py
│   ├── model.py        # Swin2SR loading, preprocessing, inference
│   └── main.py         # FastAPI + Gradio Blocks application
├── Dockerfile           # Multi-stage, HF Spaces compatible
├── requirements.txt     # Python dependencies
├── README.md
└── .gitignore
```

---

## 📄 License

MIT

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/cyash24f3">Yash Chavan</a></sub>
</p>
