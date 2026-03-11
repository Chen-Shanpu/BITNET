#!/usr/bin/env python3
"""
Extract unique weight shapes from Llama model configurations.

This script reads LlamaConfig JSON files and extracts all unique (M, K) shapes
from the linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

Usage:
    python utils/extract_model_shapes.py --config-dir model_configs/Llama-80M
    python utils/extract_model_shapes.py --all-models
"""

import argparse
import json
import os
import sys
from collections import OrderedDict


# Map of model names to config directories
MODEL_CONFIGS = {
    "Llama-80M":  "model_configs/Llama-80M",
    "Llama-120M": "model_configs/Llama-120M",
    "Llama-300M": "model_configs/Llama-300M",
    "Llama-500M": "model_configs/Llama-500M",
    "Llama-700M": "model_configs/Llama-700M",
    "Llama-1B":   "model_configs/Llama-1B",
}


def load_config(config_path):
    """Load a LlamaConfig JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_shapes_from_config(config):
    """
    Extract unique (M, K) weight shapes from a Llama model config.

    For a Llama model with:
      H = hidden_size
      I = intermediate_size
      N = num_attention_heads
      NK = num_key_value_heads
      head_dim = H / N

    The linear layer weight shapes [out_features, in_features] are:
      q_proj:    [H, H]
      k_proj:    [NK * head_dim, H]
      v_proj:    [NK * head_dim, H]
      o_proj:    [H, H]      (actually [H, N * head_dim] = [H, H])
      gate_proj: [I, H]
      up_proj:   [I, H]
      down_proj: [H, I]

    Returns:
        dict: Layer name -> (out_features, in_features)
        list: Unique [M, K] shapes for kernel tuning
    """
    H = config["hidden_size"]
    I = config["intermediate_size"]
    N = config["num_attention_heads"]
    NK = config.get("num_key_value_heads", N)
    head_dim = H // N
    kv_dim = NK * head_dim

    layer_shapes = OrderedDict()
    layer_shapes["q_proj"]    = (H, H)
    layer_shapes["k_proj"]    = (kv_dim, H)
    layer_shapes["v_proj"]    = (kv_dim, H)
    layer_shapes["o_proj"]    = (H, H)
    layer_shapes["gate_proj"] = (I, H)
    layer_shapes["up_proj"]   = (I, H)
    layer_shapes["down_proj"] = (H, I)

    # Extract unique shapes, preserving order
    seen = set()
    unique_shapes = []
    for name, shape in layer_shapes.items():
        key = (shape[0], shape[1])
        if key not in seen:
            seen.add(key)
            unique_shapes.append(list(key))

    return layer_shapes, unique_shapes


def print_model_info(model_name, config, layer_shapes, unique_shapes):
    """Print detailed model information."""
    H = config["hidden_size"]
    I = config["intermediate_size"]
    N = config["num_attention_heads"]
    NK = config.get("num_key_value_heads", N)
    L = config["num_hidden_layers"]

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"  hidden_size:         {H}")
    print(f"  intermediate_size:   {I}")
    print(f"  num_hidden_layers:   {L}")
    print(f"  num_attention_heads: {N}")
    print(f"  num_key_value_heads: {NK}")
    print(f"  head_dim:            {H // N}")

    print(f"\n  Layer weight shapes (out_features, in_features):")
    for name, shape in layer_shapes.items():
        print(f"    {name:12s}: ({shape[0]:>6d}, {shape[1]:>6d})")

    print(f"\n  Unique kernel shapes [M, K]:")
    for i, shape in enumerate(unique_shapes):
        print(f"    Shape {i}: [{shape[0]:>6d}, {shape[1]:>6d}]")

    print(f"\n  Python dict entry for codegen scripts:")
    shapes_str = ", ".join([f"[{s[0]}, {s[1]}]" for s in unique_shapes])
    print(f'    "{model_name}": [{shapes_str}]')


def main():
    parser = argparse.ArgumentParser(
        description='Extract weight shapes from Llama model configurations'
    )
    parser.add_argument('--config-dir', type=str,
                        help='Path to model config directory containing config.json')
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()),
                        help='Model name to extract shapes for')
    parser.add_argument('--all-models', action='store_true',
                        help='Extract shapes for all model configurations')
    parser.add_argument('--output-format', type=str, default='text',
                        choices=['text', 'json', 'codegen'],
                        help='Output format (default: text)')

    args = parser.parse_args()

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.all_models:
        models_to_process = MODEL_CONFIGS
    elif args.model:
        models_to_process = {args.model: MODEL_CONFIGS[args.model]}
    elif args.config_dir:
        model_name = os.path.basename(os.path.normpath(args.config_dir))
        models_to_process = {model_name: args.config_dir}
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}

    for model_name, config_dir in models_to_process.items():
        config_path = os.path.join(project_root, config_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: Config not found at {config_path}, skipping {model_name}")
            continue

        config = load_config(config_path)
        layer_shapes, unique_shapes = extract_shapes_from_config(config)
        all_results[model_name] = {
            "config": config,
            "layer_shapes": layer_shapes,
            "unique_shapes": unique_shapes,
        }

        if args.output_format == 'text':
            print_model_info(model_name, config, layer_shapes, unique_shapes)

    if args.output_format == 'json':
        output = {}
        for model_name, data in all_results.items():
            output[model_name] = {
                "unique_shapes": data["unique_shapes"],
                "layer_shapes": {k: list(v) for k, v in data["layer_shapes"].items()},
            }
        print(json.dumps(output, indent=2))

    elif args.output_format == 'codegen':
        print("\n# ModelShapeDict for codegen_tl1.py / codegen_tl2.py:")
        print("ModelShapeDict = {")
        for model_name, data in all_results.items():
            shapes = data["unique_shapes"]
            shapes_str = ",\n                                                ".join(
                [f"[{s[0]}, {s[1]}]" for s in shapes]
            )
            print(f'    "{model_name:40s}": [{shapes_str}],')
        print("}")


if __name__ == "__main__":
    main()
