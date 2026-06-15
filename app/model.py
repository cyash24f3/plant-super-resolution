"""
Super-Resolution Model Module — Swin2SR via HuggingFace Transformers.

Loads the `caidas/swin2SR-lightweight-x2-64` checkpoint for 2× image upscaling.
Provides preprocessing, GPU/CPU-aware inference, and post-processing utilities
that return a PIL Image ready for the API / Gradio UI.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME: str = "caidas/swin2SR-lightweight-x2-64"
UPSCALE_FACTOR: int = 2
MAX_INPUT_DIM: int = 1024  # clamp large uploads to prevent OOM


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
class SuperResolutionModel:
    """Thin wrapper around Swin2SR that handles device placement and caching."""

    def __init__(self) -> None:
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Loading Swin2SR model '%s' on %s …", MODEL_NAME, self._device)
        self._processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self._model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_NAME).to(
            self._device
        )
        self._model.eval()
        logger.info("Swin2SR model loaded successfully.")

    # -- public API ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def upscale_factor(self) -> int:
        return UPSCALE_FACTOR

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def upscale(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Run super-resolution on *image* and return (hr_image, metadata).

        Parameters
        ----------
        image:
            Input PIL Image (any mode — will be converted to RGB).

        Returns
        -------
        hr_image:
            Upscaled PIL Image in RGB.
        metadata:
            Dict with ``input_size``, ``output_size``, ``upscale_factor``.
        """
        image = image.convert("RGB")
        image = self._clamp_size(image)
        input_size = image.size  # (W, H)

        hr_image = self._infer(image)
        output_size = hr_image.size

        metadata = {
            "input_width": input_size[0],
            "input_height": input_size[1],
            "output_width": output_size[0],
            "output_height": output_size[1],
            "upscale_factor": f"{UPSCALE_FACTOR}×",
            "model": MODEL_NAME,
            "device": str(self._device),
        }
        return hr_image, metadata

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _clamp_size(image: Image.Image) -> Image.Image:
        """Down-scale oversized inputs so inference stays within memory."""
        w, h = image.size
        if max(w, h) > MAX_INPUT_DIM:
            ratio = MAX_INPUT_DIM / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.info("Clamped input to %d×%d", new_w, new_h)
        return image

    @staticmethod
    def _pad_to_multiple(image: Image.Image, multiple: int = 64) -> Tuple[Image.Image, Tuple[int, int]]:
        """Pad image dimensions to the nearest multiple (Swin2SR requirement)."""
        w, h = image.size
        pad_w = (multiple - w % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        if pad_w or pad_h:
            padded = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
            padded.paste(image, (0, 0))
            return padded, (w, h)
        return image, (w, h)

    @torch.inference_mode()
    def _infer(self, image: Image.Image) -> Image.Image:
        """Core inference: preprocess → forward → postprocess."""
        original_w, original_h = image.size

        # Pad to multiple of 64 for Swin2SR
        padded_image, (orig_w, orig_h) = self._pad_to_multiple(image, multiple=64)

        inputs = self._processor(padded_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        output = self._model(pixel_values)
        # output.reconstruction shape: (B, C, H, W) float32 in [0, 1]
        sr_tensor = output.reconstruction.squeeze(0).cpu().clamp(0, 1)
        sr_array = (sr_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        hr_image = Image.fromarray(sr_array)

        # Crop away padding artefacts
        target_w = orig_w * UPSCALE_FACTOR
        target_h = orig_h * UPSCALE_FACTOR
        hr_image = hr_image.crop((0, 0, target_w, target_h))

        return hr_image


# ---------------------------------------------------------------------------
# Module-level singleton accessor (lazy)
# ---------------------------------------------------------------------------
_model_instance: SuperResolutionModel | None = None


def get_model() -> SuperResolutionModel:
    """Return (and lazily initialise) the global SR model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SuperResolutionModel()
    return _model_instance
