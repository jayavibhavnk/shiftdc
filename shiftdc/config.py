from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ShiftDCConfig:
    # Model
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    dtype: str = "float16"          # "float16" | "bfloat16" | "float32"
    load_in_8bit: bool = False

    # Layers to apply ShiftDC (middle third auto-selected if None)
    intervention_layers: Optional[List[int]] = None

    # Calibration data: JSONL with {"text": "...", "label": "safe"|"unsafe"}
    calibration_data_path: str = "data/calibration.jsonl"

    # Pre-computed safety direction (saved/loaded as .pt file)
    safety_direction_path: str = "checkpoints/safety_direction.pt"

    # Captioning
    captioner_name: Optional[str] = None  # defaults to model_name
    caption_prompt: str = "Based on the request, describe the image."

    # Generation
    max_new_tokens: int = 512
    do_sample: bool = False

    # Evaluation
    benchmark: str = "mm_safety_bench"   # or "figstep"
    benchmark_data_dir: str = "data/mm_safety_bench"
    output_dir: str = "outputs"
    rejection_keywords_path: str = "data/rejection_keywords.txt"
