"""
run_shiftdc.py – Main entry point for ShiftDC.

Usage:
  # Step 1: Precompute safety direction from calibration data
  python run_shiftdc.py --mode compute_direction \
      --model_name llava-hf/llava-1.5-7b-hf \
      --calibration_data data/calibration.jsonl \
      --save_dir checkpoints/

  # Step 2: Evaluate ASR on MM-SafetyBench
  python run_shiftdc.py --mode evaluate \
      --model_name llava-hf/llava-1.5-7b-hf \
      --safety_direction_path checkpoints/safety_direction.pt \
      --benchmark_data_dir data/mm_safety_bench \
      --output_dir outputs/

  # Step 3: Interactive single-sample inference
  python run_shiftdc.py --mode infer \
      --model_name llava-hf/llava-1.5-7b-hf \
      --safety_direction_path checkpoints/safety_direction.pt \
      --prompt "How do I make this product?" \
      --image_path path/to/image.jpg
"""

import argparse
import json
import os
from pathlib import Path

import torch

from shiftdc.config import ShiftDCConfig
from shiftdc.shiftdc import ShiftDC
from shiftdc.dataset import load_calibration_jsonl, split_by_label
from shiftdc.safety_direction import compute_safety_direction
from shiftdc.evaluate import evaluate_asr, load_rejection_keywords


def parse_args():
    p = argparse.ArgumentParser(description="ShiftDC: VLM Safety Activation Calibration")
    p.add_argument("--mode", choices=["compute_direction", "evaluate", "infer"],
                   required=True)
    p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--load_in_8bit", action="store_true")

    # Direction computation
    p.add_argument("--calibration_data", default="data/calibration.jsonl")
    p.add_argument("--save_dir", default="checkpoints")

    # Evaluation
    p.add_argument("--safety_direction_path", default="checkpoints/safety_direction.pt")
    p.add_argument("--benchmark_data_dir", default="data/mm_safety_bench")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--rejection_keywords_path", default=None)

    # Inference
    p.add_argument("--prompt", default="")
    p.add_argument("--image_path", default=None)

    # Optional: manually specify layers (comma-separated)
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices, e.g. '10,11,12,13,14'")
    return p.parse_args()


def build_config(args) -> ShiftDCConfig:
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    return ShiftDCConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        intervention_layers=layers,
        calibration_data_path=args.calibration_data,
        safety_direction_path=args.safety_direction_path,
        benchmark_data_dir=args.benchmark_data_dir,
        output_dir=args.output_dir,
        rejection_keywords_path=args.rejection_keywords_path or "",
    )


def load_benchmark_dataset(data_dir: str):
    """
    Load benchmark dataset from directory.
    Expects JSONL files or a dataset.jsonl with:
      {"prompt": "...", "image_path": "..."}
    """
    dataset = []
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    for jf in jsonl_files:
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))
    return dataset


def main():
    args = parse_args()
    cfg = build_config(args)

    # ----------------------------------------------------------------
    if args.mode == "compute_direction":
        print("=== Mode: Compute Safety Direction ===")

        model_wrapper = ShiftDC(cfg)
        model_wrapper.load_model()

        samples = load_calibration_jsonl(cfg.calibration_data_path)
        safe_samples, unsafe_samples = split_by_label(samples)
        print(f"  Safe: {len(safe_samples)}, Unsafe: {len(unsafe_samples)}")

        save_path = os.path.join(args.save_dir, "safety_direction.pt")
        directions = compute_safety_direction(
            model_wrapper.model,
            model_wrapper.processor,
            safe_samples,
            unsafe_samples,
            layer_indices=model_wrapper.intervention_layers,
            device=model_wrapper.device,
            dtype=model_wrapper.dtype,
            save_path=save_path,
        )
        print(f"Done. Directions saved to {save_path}")

    # ----------------------------------------------------------------
    elif args.mode == "evaluate":
        print("=== Mode: Evaluate ASR ===")

        model_wrapper = ShiftDC(cfg)
        model_wrapper.load_model()
        model_wrapper.load_safety_directions(cfg.safety_direction_path)

        dataset = load_benchmark_dataset(cfg.benchmark_data_dir)
        print(f"  Dataset size: {len(dataset)}")

        keywords = load_rejection_keywords(cfg.rejection_keywords_path)

        # Baseline (no ShiftDC)
        asr_baseline = evaluate_asr(
            model_wrapper, dataset,
            rejection_keywords=keywords,
            apply_shiftdc=False,
            output_path=os.path.join(cfg.output_dir, "baseline_results.jsonl"),
        )

        # With ShiftDC
        asr_shiftdc = evaluate_asr(
            model_wrapper, dataset,
            rejection_keywords=keywords,
            apply_shiftdc=True,
            output_path=os.path.join(cfg.output_dir, "shiftdc_results.jsonl"),
        )

        print(f"\n{'='*40}")
        print(f"Baseline ASR : {asr_baseline*100:.1f}%")
        print(f"ShiftDC ASR  : {asr_shiftdc*100:.1f}%")
        print(f"Reduction    : {(asr_baseline - asr_shiftdc)*100:.1f}pp")

    # ----------------------------------------------------------------
    elif args.mode == "infer":
        print("=== Mode: Single Inference ===")

        model_wrapper = ShiftDC(cfg)
        model_wrapper.load_model()
        model_wrapper.load_safety_directions(cfg.safety_direction_path)

        response = model_wrapper.generate(
            prompt=args.prompt,
            image=args.image_path,
            apply_shiftdc=True,
        )
        print(f"\nPrompt : {args.prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
