"""
Plant Leaf Super-Resolution — FastAPI + Gradio Application.

Serves a polished Gradio Blocks UI at ``/`` and a REST API at ``/api/*``.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from app.model import get_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Plant Leaf Super-Resolution",
    description=(
        "Upscale low-resolution plant leaf images using a Swin2SR "
        "transformer model. Portfolio demo by Yash Chavan."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["health"])
async def health_check() -> dict:
    """Liveness / readiness probe."""
    model = get_model()
    return {
        "status": "ok",
        "model": "Swin2SR",
        "model_name": model.model_name,
        "device": str(model.device),
        "upscale_factor": model.upscale_factor,
    }


@app.post("/api/upscale", tags=["inference"])
async def api_upscale(file: UploadFile = File(...)) -> StreamingResponse:
    """Accept an image upload and return the super-resolved version as PNG."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    model = get_model()
    hr_image, meta = model.upscale(image)

    buf = io.BytesIO()
    hr_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Input-Size": f"{meta['input_width']}x{meta['input_height']}",
            "X-Output-Size": f"{meta['output_width']}x{meta['output_height']}",
            "X-Upscale-Factor": meta["upscale_factor"],
        },
    )


# ---------------------------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------------------------
def upscale_image(
    input_image: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    """Gradio callback — upscale an image and return result + info markdown."""
    if input_image is None:
        return None, "⚠️ Please upload an image first."

    pil_image = Image.fromarray(input_image)
    model = get_model()

    t0 = time.perf_counter()
    hr_image, meta = model.upscale(pil_image)
    elapsed = time.perf_counter() - t0

    info = (
        f"### ✅ Upscaling Complete\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| **Input** | {meta['input_width']} × {meta['input_height']} px |\n"
        f"| **Output** | {meta['output_width']} × {meta['output_height']} px |\n"
        f"| **Scale Factor** | {meta['upscale_factor']} |\n"
        f"| **Inference Time** | {elapsed:.2f} s |\n"
        f"| **Device** | {meta['device']} |\n"
    )

    return np.array(hr_image), info


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Root colour overrides ── */
:root {
    --primary-600: #059669;
    --primary-500: #10b981;
    --primary-400: #34d399;
    --primary-300: #6ee7b7;
    --primary-200: #a7f3d0;
    --border-color-primary: #065f46;
    --background-fill-primary: #022c22;
}

/* ── Global ── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── Header ── */
#header-block {
    text-align: center;
    padding: 1.75rem 1rem 0.75rem;
}
#header-block h1 {
    font-size: 2.2rem;
    background: linear-gradient(135deg, #34d399, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
#header-block p {
    opacity: 0.7;
    font-size: 0.95rem;
}

/* ── Image panels ── */
.image-panel {
    border: 2px solid #065f46 !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.image-panel:hover {
    border-color: #10b981 !important;
}

/* ── Run button ── */
#run-btn {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    transition: transform 0.15s, box-shadow 0.15s;
}
#run-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
}

/* ── Info panel ── */
#info-md {
    border: 1px solid #065f46;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    background: rgba(5, 150, 105, 0.06);
}

/* ── Footer ── */
#footer-block {
    text-align: center;
    padding: 1.25rem 0 0.5rem;
    opacity: 0.55;
    font-size: 0.82rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #059669; border-radius: 3px; }
"""

# ---------------------------------------------------------------------------
# Build Gradio Blocks
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    demo = gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0a0f0d",
            body_background_fill_dark="#0a0f0d",
            block_background_fill="#111916",
            block_background_fill_dark="#111916",
            block_label_text_color="#a7f3d0",
            block_label_text_color_dark="#a7f3d0",
            input_background_fill="#162118",
            input_background_fill_dark="#162118",
            body_text_color="#d1fae5",
            body_text_color_dark="#d1fae5",
            border_color_primary="#065f46",
            border_color_primary_dark="#065f46",
        ),
        title="Plant Leaf Super-Resolution",
    )

    with demo:
        # ── Header ──
        with gr.Column(elem_id="header-block"):
            gr.Markdown(
                "# 🌿 Plant Leaf Super-Resolution\n"
                "Upscale low-resolution plant leaf images using a **Swin2SR** "
                "transformer model (2× lightweight).\n\n"
                "A portfolio project by **Yash Chavan** — "
                "[GitHub](https://github.com/cyash24f3/plant-super-resolution)"
            )

        # ── Main I/O ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷  Low-Resolution Input",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    elem_classes=["image-panel"],
                    height=420,
                )
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="✨  Super-Resolved Output",
                    type="numpy",
                    interactive=False,
                    elem_classes=["image-panel"],
                    height=420,
                )

        # ── Button ──
        with gr.Row():
            run_btn = gr.Button(
                "🚀  Upscale Image",
                variant="primary",
                elem_id="run-btn",
                size="lg",
            )

        # ── Info ──
        info_md = gr.Markdown(
            "_Upload a plant leaf image and click **Upscale Image** to begin._",
            elem_id="info-md",
        )

        # ── Examples ──
        gr.Markdown("### 📂 Example Images")
        gr.Markdown(
            "_Drag & drop any plant leaf photo above, or try your own!_\n\n"
            "Tip: The model works best on images ≤ 1024 px on the longest side."
        )

        # ── Architecture ──
        with gr.Accordion("🏗️  Model Architecture Details", open=False):
            gr.Markdown(
                "The original competition model by Yash Chavan uses an "
                "**RRDB + CBAM GAN** architecture (similar to Real-ESRGAN):\n\n"
                "- **Generator**: 16 Residual-in-Residual Dense Blocks (RRDB) "
                "with Channel & Spatial Attention (CBAM)\n"
                "- **Discriminator**: VGG-style PatchGAN\n"
                "- **Losses**: Perceptual (VGG-19) + Adversarial + L1 Pixel\n\n"
                "This demo uses a **Swin2SR-Lightweight (2×)** pretrained "
                "transformer from HuggingFace as a drop-in stand-in — no "
                "custom weights needed.\n\n"
                "| Component | Detail |\n"
                "|---|---|\n"
                "| Model | `caidas/swin2SR-lightweight-x2-64` |\n"
                "| Backbone | Swin Transformer V2 |\n"
                "| Scale | 2× |\n"
                "| Parameters | ~1 M |\n"
            )

        # ── Footer ──
        gr.Markdown(
            "---\n"
            "🌱 *Portfolio demo — uses a pretrained Swin2SR model for "
            "super-resolution. Not the original competition weights.*\n\n"
            "Built with FastAPI • Gradio • PyTorch • 🤗 Transformers",
            elem_id="footer-block",
        )

        # ── Event wiring ──
        run_btn.click(
            fn=upscale_image,
            inputs=[input_image],
            outputs=[output_image, info_md],
        )
        input_image.change(
            fn=lambda _: "_Image loaded — click **Upscale Image** to run._",
            inputs=[input_image],
            outputs=[info_md],
        )

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio on FastAPI
# ---------------------------------------------------------------------------
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
