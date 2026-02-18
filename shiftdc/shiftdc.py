"""
shiftdc.py – Core ShiftDC inference-time intervention.

For a vision-language input t_vl = [p, i], ShiftDC:

  1. Generates a text caption c for image i.
  2. Runs a forward pass on text-only input t_tt = [p, c] to get x^l(t_tt).
  3. Runs a forward pass on t_vl to get x^l(t_vl).
  4. Computes modality-induced shift m^l = x^l(t_vl) - x^l(t_tt)   [Eq. 5]
  5. Projects onto safety direction:
       proj_s(m^l) = (m^l . s^l / ||s^l||^2) * s^l               [Eq. 6]
  6. Calibrated activation:
       x_hat^l(t_vl) = x^l(t_vl) - proj_s(m^l)                   [Eq. 7]
  7. Runs a final generation pass with patched activations.
"""

import torch
from typing import Dict, List, Optional

from .hooks import ActivationStore, ActivationPatcher
from .safety_direction import _get_transformer_layers


def _project_onto_direction(
    vector: torch.Tensor,   # shape: [d]
    direction: torch.Tensor # shape: [d]
) -> torch.Tensor:
    """
    Compute the projection of `vector` onto `direction`.

    proj_s(m) = (m · s / ||s||^2) * s       [Eq. 6]

    Returns tensor of same shape as vector.
    """
    direction = direction.to(vector.dtype)
    norm_sq = torch.dot(direction, direction)
    if norm_sq < 1e-10:
        return torch.zeros_like(vector)
    scalar = torch.dot(vector, direction) / norm_sq
    return scalar * direction


def _generate_caption(
    model,
    processor,
    image,
    text_prompt: str,
    caption_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 128,
) -> str:
    """
    Use the VLM itself to generate a textual caption for the image.
    Prompt: "Based on the request, describe the image."
    """
    combined_text = f"{text_prompt}\n{caption_prompt}"
    inputs = processor(text=combined_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the new tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated, skip_special_tokens=True)
    return caption.strip()


class ShiftDC:
    """
    Inference-time Activation Shift Disentanglement and Calibration.

    Usage:
        from shiftdc import ShiftDC, ShiftDCConfig, compute_safety_direction

        cfg = ShiftDCConfig(model_name="llava-hf/llava-1.5-7b-hf")
        shiftdc = ShiftDC(cfg)
        shiftdc.load_model()

        # Compute safety direction once from calibration data
        safety_dirs = compute_safety_direction(
            shiftdc.model, shiftdc.processor,
            safe_samples, unsafe_samples,
            save_path=cfg.safety_direction_path
        )
        shiftdc.set_safety_directions(safety_dirs)

        # Run inference with ShiftDC
        response = shiftdc.generate(prompt="How to make this?", image=my_pil_image)
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.safety_directions: Optional[Dict[int, torch.Tensor]] = None
        self.device = torch.device(config.device)
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[config.dtype]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_model(self):
        """Load VLM model and processor from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import transformers

        print(f"[ShiftDC] Loading model: {self.config.model_name}")

        load_kwargs = {"torch_dtype": self.dtype}
        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name, **load_kwargs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        print("[ShiftDC] Model loaded.")

        # Auto-set intervention layers if not manually specified
        if self.config.intervention_layers is None:
            layers = _get_transformer_layers(self.model)
            n = len(layers)
            start, end = n // 3, 2 * n // 3
            self.intervention_layers = list(range(start, end))
        else:
            self.intervention_layers = self.config.intervention_layers

        print(f"[ShiftDC] Intervention layers: {self.intervention_layers}")

    def set_safety_directions(self, directions: Dict[int, torch.Tensor]):
        """Set pre-computed safety directions s^l."""
        self.safety_directions = directions
        print(f"[ShiftDC] Safety directions set for {len(directions)} layers.")

    def load_safety_directions(self, path: str):
        """Load safety directions from a saved .pt file."""
        from .safety_direction import load_safety_direction
        self.safety_directions = load_safety_direction(path)
        print(f"[ShiftDC] Loaded safety directions from {path}")

    # ------------------------------------------------------------------
    # Core inference: Step-by-step per Equations 5–7
    # ------------------------------------------------------------------

    def _extract_activations(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Run one forward pass and return last-token activations per layer."""
        layers = _get_transformer_layers(self.model)
        monitored = [layers[i] for i in self.intervention_layers]
        store = ActivationStore()
        store.register(monitored, self.intervention_layers)

        with torch.no_grad():
            self.model(**inputs)

        store.clear_hooks()
        return dict(store.activations)  # {layer_idx: tensor [1, d]}

    def _compute_patches(
        self,
        act_vl: Dict[int, torch.Tensor],
        act_tt: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        For each layer, compute the patch (safety-relevant shift to remove):

            patch^l = proj_{s^l}(m^l)
                    = proj_{s^l}(x^l(t_vl) - x^l(t_tt))      [Eq. 5, 6]

        The patch is later *subtracted* from the activation during generation:
            x_hat^l = x^l(t_vl) - patch^l                     [Eq. 7]
        """
        patches = {}
        for layer_idx in self.intervention_layers:
            if layer_idx not in act_vl or layer_idx not in act_tt:
                continue
            if layer_idx not in self.safety_directions:
                continue

            x_vl = act_vl[layer_idx][0].float()   # [d]
            x_tt = act_tt[layer_idx][0].float()   # [d]
            s = self.safety_directions[layer_idx].float()  # [d]

            # m^l = x^l(t_vl) - x^l(t_tt)        [Eq. 5]
            m = x_vl - x_tt

            # proj_{s^l}(m^l)                      [Eq. 6]
            projection = _project_onto_direction(m, s)

            patches[layer_idx] = projection.unsqueeze(0)  # [1, d]

        return patches

    def generate(
        self,
        prompt: str,
        image=None,
        apply_shiftdc: bool = True,
    ) -> str:
        """
        Generate a response for the given prompt and optional image.

        If `image` is None or `apply_shiftdc` is False, performs standard
        inference without any activation intervention.

        Args:
            prompt: Text instruction.
            image: PIL.Image or path string. Pass None for text-only.
            apply_shiftdc: Whether to apply the ShiftDC intervention.

        Returns:
            Generated response string.
        """
        assert self.model is not None, "Call load_model() first."

        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        # ---- Standard text-only or ShiftDC disabled ----
        if image is None or not apply_shiftdc:
            if image is not None:
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                )
            else:
                inputs = self.processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )
            generated = out[0][inputs["input_ids"].shape[1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()

        # ---- ShiftDC: 3-pass procedure ----
        assert self.safety_directions is not None, \
            "Call set_safety_directions() or load_safety_directions() first."

        # --- Pass 1: Generate caption c for image i ---
        caption = _generate_caption(
            self.model, self.processor, image, prompt,
            self.config.caption_prompt,
            self.device, self.dtype,
        )

        # --- Pass 2a: Extract activations for t_tt = [p, c] ---
        text_only_input = f"{prompt}\n{caption}"
        inputs_tt = self.processor(text=text_only_input, return_tensors="pt")
        inputs_tt = {k: v.to(self.device) for k, v in inputs_tt.items()}
        act_tt = self._extract_activations(inputs_tt)

        # --- Pass 2b: Extract activations for t_vl = [p, i] ---
        inputs_vl = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs_vl = {k: v.to(self.device) for k, v in inputs_vl.items()}
        if "pixel_values" in inputs_vl:
            inputs_vl["pixel_values"] = inputs_vl["pixel_values"].to(self.dtype)
        act_vl = self._extract_activations(inputs_vl)

        # --- Compute per-layer patches [Eq. 5–6] ---
        patches = self._compute_patches(act_vl, act_tt)

        # --- Pass 3: Generate with activation patches applied [Eq. 7] ---
        layers = _get_transformer_layers(self.model)
        monitored = [layers[i] for i in self.intervention_layers]
        patcher = ActivationPatcher()
        patcher.register(monitored, self.intervention_layers, patches, self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs_vl,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )

        patcher.clear_hooks()

        generated = out[0][inputs_vl["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
