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
    direction = direction.to(vector.dtype)
    denom = torch.dot(direction, direction)
    if denom < 1e-10:
        return torch.zeros_like(vector)
    return (torch.dot(vector, direction) / denom) * direction


def _to_pil(image) -> PILImage.Image:
    if isinstance(image, str):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected path or PIL.Image, got {type(image)}")


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

    def load_model(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        print(f"[ShiftDC] Loading {self.config.model_name} ...")
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

        if self.intervention_layers is None:
            layers = _get_transformer_layers(self.model)
            self.intervention_layers = _auto_intervention_layers(len(layers))

        print(f"[ShiftDC] Ready. Intervention layers: {self.intervention_layers}")

    def set_safety_directions(self, d):
        self.safety_directions = d

    def load_safety_directions(self, path=None):
        self.safety_directions = load_safety_direction(
            path or self.config.safety_direction_path
        )

    def compute_and_save_directions(self, safe_samples, unsafe_samples):
        d = compute_safety_direction(
            self.model,
            self.processor,
            safe_samples,
            unsafe_samples,
            layer_indices=self.intervention_layers,
            device=next(self.model.parameters()).device,
            dtype=self.dtype,
            save_path=self.config.safety_direction_path,
        )
        self.safety_directions = d
        return d

    def _prepare_inputs(self, text, image=None):
        if image is not None:
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        else:
            inputs = self.processor(text=text, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def _extract_activations(self, inputs):
        layers = _get_transformer_layers(self.model)
        modules = [layers[i] for i in self.intervention_layers]
        store = ActivationStore()
        store.register(modules, self.intervention_layers)
        with torch.no_grad():
            self.model(**inputs)
        acts = dict(store.activations)
        store.remove()
        return acts

    def _caption_image(self, image, prompt):
        combined = prompt + "\n" + self.config.caption_prompt
        inputs = self._prepare_inputs(combined, image=image)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _compute_patches(self, act_vl, act_tt):
        patches = {}
        for idx in self.intervention_layers:
            if idx not in act_vl or idx not in act_tt:
                continue
            if self.safety_directions is None or idx not in self.safety_directions:
                continue
            m = act_vl[idx][0].float() - act_tt[idx][0].float()
            s = self.safety_directions[idx].float()
            patches[idx] = _project_onto(m, s).unsqueeze(0)
        return patches

    def generate(self, prompt, image=None, apply_shiftdc=True):
        assert self.model is not None, "Call load_model() first."
        pil = _to_pil(image) if image is not None else None

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

        assert self.safety_directions is not None, "Call load_safety_directions() first."

        caption = self._caption_image(pil, prompt)
        inputs_tt = self._prepare_inputs(prompt + "\n" + caption)
        act_tt = self._extract_activations(inputs_tt)
        inputs_vl = self._prepare_inputs(prompt, image=pil)
        act_vl = self._extract_activations(inputs_vl)
        patches = self._compute_patches(act_vl, act_tt)

        layers = _get_transformer_layers(self.model)
        modules = [layers[i] for i in self.intervention_layers]
        patcher = ActivationPatcher()
        dev = next(self.model.parameters()).device
        patcher.register(modules, self.intervention_layers, patches, device=dev)

        with torch.no_grad():
            out = self.model.generate(
                **inputs_vl,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        patcher.remove()

        new_tokens = out[0][inputs_vl["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
