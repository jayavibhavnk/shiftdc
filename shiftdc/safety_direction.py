"""
safety_direction.py

Computes the safety-relevant direction s^l for each transformer layer l:

    s^l = ActMean^l(D_safe_tt) - ActMean^l(D_unsafe_tt)       [Eq. 4]

where ActMean^l(D) = (1/|D|) * sum_{t in D} x^l(t)            [Eq. 1]

x^l(t) is the residual-stream activation of the *last token* at layer l.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from PIL import Image

from .dataset import CalibrationSample, load_image
from .hooks import ActivationStore


def _get_transformer_layers(model) -> List[torch.nn.Module]:
    """
    Retrieve the list of transformer decoder layers from common VLM architectures.
    Supports LLaVA (uses model.language_model.model.layers or model.model.layers),
    MiniGPT-4, ShareGPT4V, Qwen-VL.
    """
    # LLaVA-1.5 / LLaVA-1.6 (HuggingFace)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return list(model.language_model.model.layers)
    # Older LLaVA or Vicuna backbone
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Qwen-VL
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(
        "Cannot auto-detect transformer layers. "
        "Please subclass and override _get_transformer_layers()."
    )


def _auto_intervention_layers(num_layers: int) -> List[int]:
    """
    The paper finds safety directions are most effective in the middle layers.
    Default: middle third of all layers.
    """
    start = num_layers // 3
    end = 2 * num_layers // 3
    return list(range(start, end))


def _collect_mean_activations(
    model,
    processor,
    samples: List[CalibrationSample],
    layer_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    use_image: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Run forward passes over `samples` and compute the mean last-token
    activation at each specified layer.

    Args:
        use_image: If True, load image from sample.image_path (Dvl).
                   If False, use sample.caption as text-only input (Dtt).

    Returns:
        Dict mapping layer_idx -> mean activation tensor of shape [d_model].
    """
    layers = _get_transformer_layers(model)
    store = ActivationStore()
    monitored_layers = [layers[i] for i in layer_indices]
    store.register(monitored_layers, layer_indices)

    # Accumulate sums per layer
    sums: Dict[int, torch.Tensor] = {}
    count = 0

    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Collecting activations", leave=False):
            if use_image and sample.image_path is not None:
                image = load_image(sample.image_path)
                text = sample.text_prompt
                inputs = processor(text=text, images=image, return_tensors="pt")
            else:
                # Text-only: use caption or plain text
                text_input = sample.caption if sample.caption else sample.text_prompt
                inputs = processor(text=text_input, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Ensure correct dtype for float tensors
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            _ = model(**inputs)

            for layer_idx, act in store.activations.items():
                act = act.float()  # accumulate in fp32
                if layer_idx not in sums:
                    sums[layer_idx] = torch.zeros_like(act)
                sums[layer_idx] += act

            store.activations.clear()
            count += 1

    store.clear_hooks()

    # Divide to get mean
    return {idx: sums[idx] / count for idx in layer_indices}


def compute_safety_direction(
    model,
    processor,
    safe_samples: List[CalibrationSample],
    unsafe_samples: List[CalibrationSample],
    layer_indices: Optional[List[int]] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
    save_path: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute s^l for each layer l (Eq. 4):

        s^l = ActMean^l(D_safe_tt) - ActMean^l(D_unsafe_tt)

    Both safe_samples and unsafe_samples should be *text-only* inputs
    (captions or plain text prompts, no images) for maximum linear
    separability as described in Section 4 Observation 1.

    Args:
        model: The VLM model (HuggingFace).
        processor: The VLM processor.
        safe_samples: List of CalibrationSample with label="safe".
        unsafe_samples: List of CalibrationSample with label="unsafe".
        layer_indices: Which layers to compute directions for.
                       Auto-selected (middle third) if None.
        device: Torch device.
        dtype: Model dtype.
        save_path: If set, saves the direction dict as a .pt file.

    Returns:
        Dict[layer_idx -> direction_tensor] of shape [d_model] each.
    """
    if device is None:
        device = next(model.parameters()).device

    layers = _get_transformer_layers(model)
    if layer_indices is None:
        layer_indices = _auto_intervention_layers(len(layers))

    print(f"[ShiftDC] Computing safety direction on {len(layer_indices)} layers "
          f"({len(safe_samples)} safe, {len(unsafe_samples)} unsafe samples)...")

    mean_safe = _collect_mean_activations(
        model, processor, safe_samples, layer_indices,
        device, dtype, use_image=False
    )
    mean_unsafe = _collect_mean_activations(
        model, processor, unsafe_samples, layer_indices,
        device, dtype, use_image=False
    )

    # s^l = mean_safe - mean_unsafe  (Eq. 4)
    safety_directions: Dict[int, torch.Tensor] = {}
    for idx in layer_indices:
        safety_directions[idx] = mean_safe[idx] - mean_unsafe[idx]   # shape: [d_model]

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(safety_directions, save_path)
        print(f"[ShiftDC] Safety directions saved to {save_path}")

    return safety_directions


def load_safety_direction(path: str) -> Dict[int, torch.Tensor]:
    return torch.load(path, map_location="cpu")
