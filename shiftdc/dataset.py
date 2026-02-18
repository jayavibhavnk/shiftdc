"""
dataset.py – Utilities for building the paired (Dvl, Dtt) calibration
             datasets described in Section 3 of the paper.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image


@dataclass
class CalibrationSample:
    text_prompt: str          # The natural-language instruction
    label: str                # "safe" or "unsafe"
    image_path: Optional[str] = None   # Path to image (for Dvl samples)
    caption: Optional[str] = None      # Text caption replacing the image


def load_calibration_jsonl(path: str) -> List[CalibrationSample]:
    """
    Load calibration data from a JSONL file.

    Expected JSONL format (one JSON object per line):
      {"text": "How to make a bomb?", "label": "unsafe", "image_path": "imgs/bomb.jpg"}
      {"text": "What is in this picture?", "label": "safe", "image_path": "imgs/cat.jpg"}
      {"text": "Write a poem about spring.", "label": "safe"}   # text-only OK too
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(CalibrationSample(
                text_prompt=obj["text"],
                label=obj["label"],
                image_path=obj.get("image_path"),
                caption=obj.get("caption"),
            ))
    return samples


def split_by_label(
    samples: List[CalibrationSample],
) -> Tuple[List[CalibrationSample], List[CalibrationSample]]:
    """Return (safe_samples, unsafe_samples)."""
    safe = [s for s in samples if s.label == "safe"]
    unsafe = [s for s in samples if s.label == "unsafe"]
    return safe, unsafe


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
