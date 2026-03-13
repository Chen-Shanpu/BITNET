#!/usr/bin/env python3
"""
BitNet Kernel Tuning Pipeline

Builds each LLaMA model from config, extracts linear layer shapes,
tunes kernel parameters for each unique shape, saves tuned kernel
configs, and produces a markdown summary report.

Usage:
    python tune_all_models.py
    python tune_all_models.py --models Llama-80M Llama-120M
    python tune_all_models.py --output-dir tuned_kernels --report tuning_report.md
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from configparser import ConfigParser
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Model config directory map
# ---------------------------------------------------------------------------
MODEL_CONFIGS = OrderedDict([
    ("Llama-80M",  "model_configs/Llama-80M"),
    ("Llama-120M", "model_configs/Llama-120M"),
    ("Llama-300M", "model_configs/Llama-300M"),
    ("Llama-500M", "model_configs/Llama-500M"),
    ("Llama-700M", "model_configs/Llama-700M"),
    ("Llama-1B",   "model_configs/Llama-1B"),
])

# ---------------------------------------------------------------------------
# Helpers – import existing tuning utilities
# ---------------------------------------------------------------------------
# Ensure the utils package is importable regardless of CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from utils.kernel_tuner import (
    generate_configs_for_shape,
    select_default_config,
    write_preset_config,
    benchmark_candidates,
    save_tuning_log,
    save_tuning_log_csv,
    select_best_from_benchmark,
)


# ---------------------------------------------------------------------------
# Step 1 & 2 – Build model, extract linear-layer shapes
# ---------------------------------------------------------------------------

def extract_shapes_from_model(config_path):
    """
    Load a LlamaConfig, instantiate the model via ``from_config()``,
    traverse all ``torch.nn.Linear`` layers and return unique (M, K) shapes.

    Returns
    -------
    layer_details : list[tuple[str, int, int]]
        (layer_name, out_features, in_features) for every Linear module.
    unique_shapes : list[list[int]]
        Deduplicated [[M, K], …] in discovery order.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    with open(config_path, "r") as fh:
        cfg_dict = json.load(fh)

    config = LlamaConfig(**cfg_dict)
    model = LlamaForCausalLM(config)

    layer_details = []
    seen = set()
    unique_shapes = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            M = module.out_features
            K = module.in_features
            layer_details.append((name, M, K))
            key = (M, K)
            if key not in seen:
                seen.add(key)
                unique_shapes.append([M, K])

    # Free memory – models can be large
    del model
    return layer_details, unique_shapes


# ---------------------------------------------------------------------------
# Step 3 – Kernel parameter tuning
# ---------------------------------------------------------------------------

def tune_shapes(unique_shapes, arch):
    """
    Tune kernel parameters for every (M, K) shape on a given architecture.

    Returns
    -------
    list[dict]
        One entry per shape with keys:
        M, K, BM, BK, bmm, num_candidates, all_configs
    """
    results = []
    for M, K in unique_shapes:
        all_configs = generate_configs_for_shape(M, K, arch)
        default = select_default_config(M, K, arch, all_configs)
        results.append({
            "M": M,
            "K": K,
            "default_BM": default[0] if default else None,
            "default_BK": default[1] if default else None,
            "default_bmm": default[2] if default else None,
            "num_candidates": len(all_configs),
        })
    return results


# ---------------------------------------------------------------------------
# Step 4 – Save tuning results to tuned_kernels/
# ---------------------------------------------------------------------------

def save_tuned_config(model_name, arch, results, output_dir):
    """Write an INI kernel config file for *arch* under *output_dir*."""
    cfg = ConfigParser()
    for i, r in enumerate(results):
        section = f"Kernels_{i}"
        cfg.add_section(section)
        cfg.set(section, "m", str(r["M"]))
        cfg.set(section, "k", str(r["K"]))
        cfg.set(section, "bm", str(r["default_BM"]))
        cfg.set(section, "bk", str(r["default_BK"]))
        cfg.set(section, "bmm", str(r["default_bmm"]))

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, f"kernel_config_{arch}.ini")
    with open(filepath, "w") as fh:
        cfg.write(fh)
    return filepath


# ---------------------------------------------------------------------------
# Step 5 – Markdown summary report
# ---------------------------------------------------------------------------

def generate_report(all_model_data, report_path):
    """
    Write a Markdown document summarising the tuning results for every model
    and architecture.

    Parameters
    ----------
    all_model_data : dict
        model_name -> {
            "config": dict,
            "layer_details": [...],
            "unique_shapes": [[M,K], ...],
            "tl1_results": [...],
            "tl2_results": [...],
            "tl1_config_path": str,
            "tl2_config_path": str,
        }
    """
    lines = []

    lines.append("# BitNet Kernel Tuning Report")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append("")

    # ---- Table of contents --------------------------------------------------
    lines.append("## Models")
    lines.append("")
    for model_name in all_model_data:
        anchor = model_name.lower().replace(" ", "-")
        lines.append(f"- [{model_name}](#{anchor})")
    lines.append("")

    # ---- Per-model sections -------------------------------------------------
    for model_name, data in all_model_data.items():
        cfg = data["config"]
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("### Model Configuration")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| hidden_size | {cfg['hidden_size']} |")
        lines.append(f"| intermediate_size | {cfg['intermediate_size']} |")
        lines.append(f"| num_hidden_layers | {cfg['num_hidden_layers']} |")
        lines.append(f"| num_attention_heads | {cfg['num_attention_heads']} |")
        lines.append(
            f"| num_key_value_heads | {cfg.get('num_key_value_heads', cfg['num_attention_heads'])} |"
        )
        lines.append(f"| vocab_size | {cfg['vocab_size']} |")
        lines.append("")

        # Unique shapes
        lines.append("### Unique Linear Layer Shapes")
        lines.append("")
        lines.append("| # | M (out) | K (in) |")
        lines.append("|---|---------|--------|")
        for idx, (M, K) in enumerate(data["unique_shapes"]):
            lines.append(f"| {idx} | {M} | {K} |")
        lines.append("")

        benchmarked = data.get("benchmarked", False)

        def _results_table(arch_label, results, config_path):
            tl_lines = []
            tl_lines.append(f"### {arch_label} Tuning Results")
            if benchmarked:
                tl_lines.append("*(parameters selected from benchmark measurements)*")
            tl_lines.append("")
            if benchmarked:
                tl_lines.append("| M | K | BM | BK | bm | Candidates | tokens/s |")
                tl_lines.append("|---|---|----|----|-----|------------|----------|")
            else:
                tl_lines.append("| M | K | BM | BK | bm | Candidates |")
                tl_lines.append("|---|---|----|----|-----|------------|")
            for r in results:
                bm_str = str(r["default_BM"]) if r["default_BM"] is not None else "—"
                bk_str = str(r["default_BK"]) if r["default_BK"] is not None else "—"
                bmm_str = str(r["default_bmm"]) if r["default_bmm"] is not None else "—"
                if benchmarked and "tokens_per_second" in r:
                    tps_str = f"{r['tokens_per_second']:.1f}"
                    tl_lines.append(
                        f"| {r['M']} | {r['K']} | {bm_str} | {bk_str} | {bmm_str}"
                        f" | {r['num_candidates']} | {tps_str} |"
                    )
                else:
                    tl_lines.append(
                        f"| {r['M']} | {r['K']} | {bm_str} | {bk_str} | {bmm_str}"
                        f" | {r['num_candidates']} |"
                    )
            tl_lines.append("")
            tl_lines.append(f"Config saved to: `{config_path}`")
            tl_lines.append("")
            return tl_lines

        lines.extend(
            _results_table(
                "TL1 (ARM NEON)", data["tl1_results"], data["tl1_config_path"]
            )
        )
        lines.extend(
            _results_table(
                "TL2 (x86 AVX2)", data["tl2_results"], data["tl2_config_path"]
            )
        )
        lines.append("---")
        lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write(report_text)
    return report_text


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BitNet kernel tuning pipeline for multiple LLaMA model scales",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to process (default: all six scales)",
    )
    parser.add_argument(
        "--output-dir",
        default="tuned_kernels",
        help="Directory to save tuned kernel configs (default: tuned_kernels)",
    )
    parser.add_argument(
        "--report",
        default="tuning_report.md",
        help="Path for the Markdown summary report (default: tuning_report.md)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks for every candidate configuration and record results",
    )
    parser.add_argument(
        "--tuning-logs-dir",
        default="tuning_logs",
        help="Directory to save full tuning logs (default: tuning_logs)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for benchmarking (default: 128)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of forward passes for benchmarking (default: 50)",
    )
    args = parser.parse_args()

    project_root = _SCRIPT_DIR
    all_model_data = OrderedDict()

    for model_name in args.models:
        config_dir = MODEL_CONFIGS[model_name]
        config_path = os.path.join(project_root, config_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"⚠  Config not found: {config_path} – skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        # -- Step 1 & 2: build model, extract shapes -------------------------
        print(f"  Building model from {config_path} …")
        layer_details, unique_shapes = extract_shapes_from_model(config_path)
        print(f"  Found {len(layer_details)} Linear layers, "
              f"{len(unique_shapes)} unique shapes")
        for M, K in unique_shapes:
            print(f"    (M={M}, K={K})")

        with open(config_path, "r") as fh:
            config_dict = json.load(fh)

        # -- Step 3 & 4: kernel tuning + save configs -------------------------
        output_dir = os.path.join(project_root, args.output_dir)

        if args.benchmark:
            # Benchmark-driven tuning: measure all candidate configurations,
            # select the best per shape, then write the optimal kernel configs.
            logs_dir = os.path.join(project_root, args.tuning_logs_dir)
            tl1_results = None
            tl2_results = None

            for arch_label, arch_tag in [("TL1", "tl1"), ("TL2", "tl2")]:
                print(f"  Benchmarking {arch_label} candidates …")
                candidates = benchmark_candidates(
                    model_name, config_path, unique_shapes, arch_tag,
                    sequence_length=args.seq_len,
                    batch_size=1,
                    num_runs=args.num_runs,
                )

                # Persist full log (JSON) and ranked CSV summary
                log_path = save_tuning_log(model_name, arch_tag, candidates, logs_dir)
                csv_path = save_tuning_log_csv(model_name, arch_tag, candidates, logs_dir)
                print(f"    Logged {len(candidates)} candidates → {log_path}")
                print(f"    CSV summary → {csv_path}")

                # Pick the best config per shape from the benchmark results
                best_results = select_best_from_benchmark(candidates)

                if arch_tag == "tl1":
                    tl1_results = best_results
                else:
                    tl2_results = best_results

                # Show the best candidate per shape
                for r in best_results:
                    print(
                        f"    Shape (M={r['M']}, K={r['K']}): best "
                        f"BM={r['default_BM']}, BK={r['default_BK']}, "
                        f"bm={r['default_bmm']}  "
                        f"({r['tokens_per_second']:.1f} tokens/s, "
                        f"{r['num_candidates']} candidates)"
                    )

            # Write benchmark-optimal kernel configs to tuned_kernels/
            tl1_path = save_tuned_config(model_name, "tl1", tl1_results, output_dir)
            tl2_path = save_tuned_config(model_name, "tl2", tl2_results, output_dir)
            print(f"  Saved TL1 config (benchmark-optimal) → {tl1_path}")
            print(f"  Saved TL2 config (benchmark-optimal) → {tl2_path}")

            # Also update the preset_kernels/ directory so the runtime picks up
            # the benchmark-optimal parameters immediately.
            preset_dir = os.path.join(project_root, "preset_kernels", model_name)
            write_preset_config(model_name, "tl1", tl1_results, preset_dir)
            write_preset_config(model_name, "tl2", tl2_results, preset_dir)
            print(f"  Updated preset_kernels/{model_name}/ with benchmark-optimal configs")

            benchmarked = True

        else:
            # Heuristic tuning: fast path using scoring heuristics (no benchmarks).
            print("  Tuning TL1 (ARM NEON) …")
            tl1_results = tune_shapes(unique_shapes, "tl1")
            for r in tl1_results:
                print(f"    M={r['M']}, K={r['K']}  →  "
                      f"BM={r['default_BM']}, BK={r['default_BK']}, "
                      f"bm={r['default_bmm']}  ({r['num_candidates']} candidates)")

            print("  Tuning TL2 (x86 AVX2) …")
            tl2_results = tune_shapes(unique_shapes, "tl2")
            for r in tl2_results:
                print(f"    M={r['M']}, K={r['K']}  →  "
                      f"BM={r['default_BM']}, BK={r['default_BK']}, "
                      f"bm={r['default_bmm']}  ({r['num_candidates']} candidates)")

            tl1_path = save_tuned_config(model_name, "tl1", tl1_results, output_dir)
            tl2_path = save_tuned_config(model_name, "tl2", tl2_results, output_dir)
            print(f"  Saved TL1 config → {tl1_path}")
            print(f"  Saved TL2 config → {tl2_path}")

            benchmarked = False

        all_model_data[model_name] = {
            "config": config_dict,
            "layer_details": layer_details,
            "unique_shapes": unique_shapes,
            "tl1_results": tl1_results,
            "tl2_results": tl2_results,
            "tl1_config_path": os.path.relpath(tl1_path, project_root),
            "tl2_config_path": os.path.relpath(tl2_path, project_root),
            "benchmarked": benchmarked,
        }


    # -- Step 5: generate report ----------------------------------------------
    report_path = os.path.join(project_root, args.report)
    generate_report(all_model_data, report_path)
    print(f"\n✅ Tuning report written to {report_path}")


if __name__ == "__main__":
    main()
