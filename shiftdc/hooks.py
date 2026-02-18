"""
hooks.py – Forward hook utilities for extracting and patching
           last-token residual-stream activations from VLM layers.
"""

import torch
from typing import Dict, List, Optional


class ActivationStore:
    """Stores last-token activations for specified layer indices."""

    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles = []

    def register(self, layers: List[torch.nn.Module], layer_indices: List[int]):
        """
        Attach hooks to the given layer modules.

        Args:
            layers: List of nn.Module objects (one per transformer layer).
            layer_indices: Corresponding integer indices for bookkeeping.
        """
        self.clear_hooks()
        for idx, layer in zip(layer_indices, layers):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output is typically a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Store last-token activation, detached, on CPU to save VRAM
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def clear_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.activations = {}


class ActivationPatcher:
    """
    Patches residual-stream activations at specified layers during a
    forward pass. Used by ShiftDC to subtract the safety-relevant shift.
    """

    def __init__(self):
        self._handles = []

    def register(
        self,
        layers: List[torch.nn.Module],
        layer_indices: List[int],
        patches: Dict[int, torch.Tensor],
        device: torch.device,
    ):
        """
        Args:
            layers: nn.Module objects for each transformer layer.
            layer_indices: Integer layer indices.
            patches: Dict mapping layer_idx -> delta tensor (shape: [1, d]).
                     This delta will be *subtracted* from the last-token
                     activation, i.e. x_hat = x - proj_s(m).
            device: Target device for the patch tensors.
        """
        self.clear_hooks()
        for idx, layer in zip(layer_indices, layers):
            if idx in patches:
                handle = layer.register_forward_hook(
                    self._make_patch_hook(patches[idx].to(device))
                )
                self._handles.append(handle)

    def _make_patch_hook(self, delta: torch.Tensor):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Subtract safety-relevant shift from last token only
            hidden[:, -1, :] = hidden[:, -1, :] - delta.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    def clear_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
