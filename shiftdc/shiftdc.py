
"""
shiftdc/shiftdc.py – ShiftDC core implementation (fixed for Colab).

Fixes:
- No AutoModelForVision2Seq dependency.
- Uses underscore helper names from safety_direction.py:
    _get_transformer_layers, _auto_intervention_layers
- Loads LLaVA via LlavaForConditionalGeneration (HF Transformers).
"""

from typing import Dict, List, Optional, Union

import torch
from PIL import Image as PILImage

from .config import ShiftDCConfig
from .hooks import ActivationStore, ActivationPatcher
from .safety_direction import (
    _get_transformer_layers,
    _auto_intervention_layers,
    compute_safety_direction,
    load_safety_direction,
)


def _project_onto(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    proj_s(m) = (m·s / ||s||^2) * s     [Paper Eq. 6]
    """
    direction = direction.to(vector.dtype)
    denom = torch.dot(direction, direction)
    if denom < 1e-10:
        return torch.zeros_like(vector)
    return (torch.dot(vector, direction) / denom) * direction


def _to_pil(image: Union[str, PILImage.Image]) -> PILImage.Image:
    if isinstance(image, str):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    raise TypeError(f"image must be a path or PIL.Image, got {type(image)}")


class ShiftDC:
    def __init__(self, config: ShiftDCConfig):
        self.config = config
        self.model = None
        self.processor = None

        self.device = torch.device(config.device)
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[config.dtype]

        self.intervention_layers: Optional[List[int]] = config.intervention_layers
        self.safety_directions: Optional[Dict[int, torch.Tensor]] = None

    # ──────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────
    def load_model(self):
        """
        Load LLaVA using Transformers. This avoids AutoModelForVision2Seq which
        may not exist in your installed transformers.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print(f"[ShiftDC] Loading model: {self.config.model_name}")

        kwargs = {"torch_dtype": self.dtype}
        if self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["device_map"] = "auto"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_name, **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

        # Set intervention layers if not given: middle third
        if self.intervention_layers is None:
            layers = _get_transformer_layers(self.model)
            self.intervention_layers = _auto_intervention_layers(len(layers))

        print("[ShiftDC] Model loaded.")
        print(f"[ShiftDC] Intervention layers: {self.intervention_layers}")

    # ──────────────────────────────────────────────────────────────────
    # Safety direction management
    # ──────────────────────────────────────────────────────────────────
    def set_safety_directions(self, directions: Dict[int, torch.Tensor]):
        self.safety_directions = directions

    def load_safety_directions(self, path: Optional[str] = None):
        p = path or self.config.safety_direction_path
        self.safety_directions = load_safety_direction(p)

    def compute_and_save_directions(self, safe_samples, unsafe_samples):
        assert self.model is not None and self.processor is not None, "Call load_model() first."
        directions = compute_safety_direction(
            self.model,
            self.processor,
            safe_samples,
            unsafe_samples,
            layer_indices=self.intervention_layers,
            device=next(self.model.parameters()).device,
            dtype=self.dtype,
            save_path=self.config.safety_direction_path,
        )
        self.safety_directions = directions
        return directions

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────
    def _prepare_inputs(self, text: str, image: Optional[PILImage.Image] = None) -> Dict[str, torch.Tensor]:
        if image is not None:
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        else:
            inputs = self.processor(text=text, return_tensors="pt")

        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        return inputs

    def _extract_activations(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        layers = _get_transformer_layers(self.model)
        modules = [layers[i] for i in self.intervention_layers]
        store = ActivationStore()
        store.register(modules, self.intervention_layers)

        with torch.no_grad():
            self.model(**inputs)

        acts = dict(store.activations)
        store.remove()
        return acts

    def _caption_image(self, image: PILImage.Image, prompt: str) -> str:
        """
        Paper uses a caption c to form t_tt=[p,c].
        """
        combined = f"{prompt}\n{self.config.caption_prompt}"
        inputs = self._prepare_inputs(combined, image=image)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=getattr(self.config, "caption_max_tokens", 128),
                do_sample=False,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return caption

    def _compute_patches(self, act_vl: Dict[int, torch.Tensor], act_tt: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        m^l = x^l(t_vl) - x^l(t_tt)           [Eq. 5]
        patch^l = proj_{s^l}(m^l)             [Eq. 6]
        """
        patches: Dict[int, torch.Tensor] = {}
        for idx in self.intervention_layers:
            if idx not in act_vl or idx not in act_tt:
                continue
            if self.safety_directions is None or idx not in self.safety_directions:
                continue

            x_vl = act_vl[idx][0].float()
            x_tt = act_tt[idx][0].float()
            s = self.safety_directions[idx].float()

            m = x_vl - x_tt
            patches[idx] = _project_onto(m, s).unsqueeze(0)  # [1, d]
        return patches

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def generate(self, prompt: str, image=None, apply_shiftdc: bool = True) -> str:
        assert self.model is not None and self.processor is not None, "Call load_model() first."

        pil = _to_pil(image) if image is not None else None

        # Baseline path
        if pil is None or not apply_shiftdc:
            inputs = self._prepare_inputs(prompt, image=pil)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        # ShiftDC path
        if self.safety_directions is None:
            raise RuntimeError("Safety directions not loaded. Call load_safety_directions() or compute_and_save_directions().")

        # Pass 1: caption
        caption = self._caption_image(pil, prompt)

        # Pass 2a: text-only activations
        inputs_tt = self._prepare_inputs(f"{prompt}\n{caption}", image=None)
        act_tt = self._extract_activations(inputs_tt)

        # Pass 2b: VL activations
        inputs_vl = self._prepare_inputs(prompt, image=pil)
        act_vl = self._extract_activations(inputs_vl)

        patches = self._compute_patches(act_vl, act_tt)

        # Pass 3: generate with patched activations (Eq. 7)
        layers = _get_transformer_layers(self.model)
        modules = [layers[i] for i in self.intervention_layers]
        patcher = ActivationPatcher()
        patcher.register(modules, self.intervention_layers, patches, device=next(self.model.parameters()).device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs_vl,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        patcher.remove()

        new_tokens = out[0][inputs_vl["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
