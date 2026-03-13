#!/usr/bin/env python3
"""
BitNet Kernel Parameter Tuner

Automated tuning script for BitNet kernel parameters (BM, BK, bm) across
multiple Llama model scales. Generates valid parameter combinations, creates
codegen commands, and produces preset kernel configurations.

Tuning Parameters:
    BM  - Block M size: rows per tile when splitting weight matrix (M, K) into (BM, K) tiles
    BK  - Block K size: columns per tile when splitting weight matrix (M, K) into (M, BK) tiles
    bm  - Sub-block size: SIMD computation unit within each BM block

Constraints:
    TL1 (ARM NEON):
        - M % BM == 0  (BM must divide M)
        - K % BK == 0  (BK must divide K)
        - bm in {32, 64}
        - BM % bm == 0  (BM must be divisible by bm)

    TL2 (x86 AVX2):
        - M % BM == 0  (BM must divide M)
        - (K % BK) % 32 == 0  (remainder of K/BK must be multiple of 32)
        - bm in {32}
        - BM % bm == 0  (BM must be divisible by bm)

Usage:
    # Generate default tuning configs for all models
    python utils/kernel_tuner.py --all-models --arch tl1

    # Generate configs for a specific model
    python utils/kernel_tuner.py --model Llama-80M --arch tl2

    # Search with custom BM/BK ranges
    python utils/kernel_tuner.py --model Llama-300M --arch tl1 --bm-range 64,128,256 --bk-range 32,64,128

    # Generate codegen commands
    python utils/kernel_tuner.py --all-models --arch tl1 --generate-commands

    # Write preset kernel configs
    python utils/kernel_tuner.py --all-models --arch tl1 --write-presets
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from configparser import ConfigParser
from datetime import datetime, timezone
from itertools import product


# =============================================================================
# Model Shape Definitions
# =============================================================================

# Unique [M, K] kernel shapes for each model
# Format: model_name -> [[M1, K1], [M2, K2], ...]
MODEL_SHAPE_DICT = {
    # Existing models
    "bitnet_b1_58-large":         [[1536, 4096],
                                   [1536, 1536],
                                   [4096, 1536]],
    "bitnet_b1_58-3B":            [[3200, 8640],
                                   [3200, 3200],
                                   [8640, 3200]],
    "Llama3-8B-1.58-100B-tokens": [[14336, 4096],
                                   [4096, 14336],
                                   [1024, 4096],
                                   [4096, 4096]],
    # New model scales
    "Llama-80M":                  [[512, 1408],
                                   [512, 512],
                                   [1408, 512]],
    "Llama-120M":                 [[768, 2048],
                                   [768, 768],
                                   [2048, 768]],
    "Llama-300M":                 [[1024, 2816],
                                   [1024, 1024],
                                   [2816, 1024]],
    "Llama-500M":                 [[1280, 3456],
                                   [1280, 1280],
                                   [3456, 1280]],
    "Llama-700M":                 [[1536, 4096],
                                   [1536, 1536],
                                   [4096, 1536]],
    "Llama-1B":                   [[2048, 5632],
                                   [2048, 2048],
                                   [5632, 2048]],
}

# Standard candidate values for parameter search
DEFAULT_BM_CANDIDATES = [32, 64, 128, 160, 192, 256, 320, 384, 512]
DEFAULT_BK_CANDIDATES_TL1 = [32, 64, 128, 256]
DEFAULT_BK_CANDIDATES_TL2 = [96, 192]
DEFAULT_BMM_CANDIDATES_TL1 = [32, 64]
DEFAULT_BMM_CANDIDATES_TL2 = [32]


# =============================================================================
# Parameter Validation
# =============================================================================

def get_valid_bm_values(M, bm_candidates, bmm_values):
    """Get valid BM values for a given M dimension."""
    valid = []
    for bm in bm_candidates:
        if M % bm == 0 and bm <= M:
            # Check that BM is divisible by at least one bmm
            if any(bm % bmm == 0 for bmm in bmm_values):
                valid.append(bm)
    return valid


def get_valid_bk_values_tl1(K, bk_candidates):
    """Get valid BK values for TL1 (ARM): K % BK == 0."""
    return [bk for bk in bk_candidates if K % bk == 0 and bk <= K]


def get_valid_bk_values_tl2(K, bk_candidates):
    """Get valid BK values for TL2 (x86): (K % BK) % 32 == 0."""
    return [bk for bk in bk_candidates if (K % bk) % 32 == 0 and bk <= K]


def get_valid_bmm_values(BM, bmm_candidates):
    """Get valid bmm (sub-block) values: BM % bmm == 0."""
    return [bmm for bmm in bmm_candidates if BM % bmm == 0]


# =============================================================================
# Configuration Generation
# =============================================================================

def generate_configs_for_shape(M, K, arch, bm_range=None, bk_range=None):
    """
    Generate all valid (BM, BK, bmm) configurations for a single (M, K) shape.

    Args:
        M: Output dimension (rows)
        K: Input dimension (columns)
        arch: Architecture type ('tl1' or 'tl2')
        bm_range: Custom BM candidates (optional)
        bk_range: Custom BK candidates (optional)

    Returns:
        List of (BM, BK, bmm) tuples
    """
    if arch == 'tl1':
        bmm_candidates = DEFAULT_BMM_CANDIDATES_TL1
        bk_candidates = bk_range or DEFAULT_BK_CANDIDATES_TL1
        bm_candidates = bm_range or DEFAULT_BM_CANDIDATES
    elif arch == 'tl2':
        bmm_candidates = DEFAULT_BMM_CANDIDATES_TL2
        bk_candidates = bk_range or DEFAULT_BK_CANDIDATES_TL2
        bm_candidates = bm_range or DEFAULT_BM_CANDIDATES
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    valid_bm = get_valid_bm_values(M, bm_candidates, bmm_candidates)

    if arch == 'tl1':
        valid_bk = get_valid_bk_values_tl1(K, bk_candidates)
    else:
        valid_bk = get_valid_bk_values_tl2(K, bk_candidates)

    configs = []
    for bm_val, bk_val in product(valid_bm, valid_bk):
        valid_bmm = get_valid_bmm_values(bm_val, bmm_candidates)
        for bmm_val in valid_bmm:
            configs.append((bm_val, bk_val, bmm_val))

    return configs


def select_default_config(M, K, arch, configs):
    """
    Select a reasonable default configuration using heuristics.

    Heuristics based on analysis of existing preset kernel configs:
    - Prefer BM in range [128, 256] for good cache utilization
    - For TL2, prefer BK=96
    - For TL1, prefer BK=64 or 128 depending on K size
    - Prefer bmm=32 for larger BM values, bmm=64 for smaller ones
    """
    if not configs:
        return None

    # Scoring thresholds based on analysis of existing BitNet preset configs
    # BM values in [128, 256] provide the best cache-to-compute balance
    BM_OPTIMAL_MIN = 128
    BM_OPTIMAL_MAX = 256
    BM_ACCEPTABLE_MIN = 64
    BM_ACCEPTABLE_MAX = 320
    # Number of M-tiles in [4, 16] keeps parallelism efficient
    TILES_OPTIMAL_MIN = 4
    TILES_OPTIMAL_MAX = 16
    TILES_ACCEPTABLE_MIN = 2
    TILES_ACCEPTABLE_MAX = 32
    # K threshold for choosing between BK=64 and BK=128 on TL1
    K_LARGE_THRESHOLD = 2048

    def score(config):
        bm_val, bk_val, bmm_val = config
        s = 0

        # Prefer BM in optimal range
        if BM_OPTIMAL_MIN <= bm_val <= BM_OPTIMAL_MAX:
            s += 10
        elif BM_ACCEPTABLE_MIN <= bm_val <= BM_ACCEPTABLE_MAX:
            s += 5

        # Prefer moderate number of tiles
        n_tiles_m = M // bm_val
        if TILES_OPTIMAL_MIN <= n_tiles_m <= TILES_OPTIMAL_MAX:
            s += 5
        elif TILES_ACCEPTABLE_MIN <= n_tiles_m <= TILES_ACCEPTABLE_MAX:
            s += 2

        if arch == 'tl1':
            # For TL1, prefer BK=128 for large K, BK=64 for small K
            if K >= K_LARGE_THRESHOLD and bk_val >= 128:
                s += 5
            elif K < K_LARGE_THRESHOLD and bk_val == 64:
                s += 5
            # Prefer bmm=64 for smaller BM, bmm=32 for larger BM
            if bm_val <= BM_OPTIMAL_MIN and bmm_val == 64:
                s += 3
            elif bm_val > BM_OPTIMAL_MIN and bmm_val == 32:
                s += 3
        else:  # tl2
            # For TL2, BK=96 is the standard
            if bk_val == 96:
                s += 5
            # bmm is always 32 for TL2
            if bmm_val == 32:
                s += 5

        return s

    return max(configs, key=score)


def generate_model_configs(model_name, arch, bm_range=None, bk_range=None):
    """
    Generate kernel configurations for all shapes of a model.

    Returns:
        List of dicts with keys: M, K, BM, BK, bmm, all_configs
    """
    shapes = MODEL_SHAPE_DICT.get(model_name)
    if shapes is None:
        raise ValueError(f"Unknown model: {model_name}")

    results = []
    for M, K in shapes:
        all_configs = generate_configs_for_shape(M, K, arch, bm_range, bk_range)
        default = select_default_config(M, K, arch, all_configs)

        results.append({
            'M': M,
            'K': K,
            'all_configs': all_configs,
            'default_BM': default[0] if default else None,
            'default_BK': default[1] if default else None,
            'default_bmm': default[2] if default else None,
            'num_candidates': len(all_configs),
        })

    return results


# =============================================================================
# Output Formatters
# =============================================================================

def print_shape_analysis(model_name, arch, results):
    """Print detailed analysis of parameter candidates for each shape."""
    print(f"\n{'='*80}")
    print(f"Kernel Parameter Analysis: {model_name} ({arch.upper()})")
    print(f"{'='*80}")

    for i, r in enumerate(results):
        print(f"\n  Shape {i}: M={r['M']}, K={r['K']}")
        print(f"  {'─'*60}")
        print(f"  Valid configurations: {r['num_candidates']}")
        if r['default_BM'] is not None:
            print(f"  ★ Default: BM={r['default_BM']}, BK={r['default_BK']}, bmm={r['default_bmm']}")

        if r['num_candidates'] <= 20:
            print(f"  All candidates:")
            for bm, bk, bmm in r['all_configs']:
                marker = " ★" if (bm, bk, bmm) == (r['default_BM'], r['default_BK'], r['default_bmm']) else ""
                print(f"    BM={bm:>4d}, BK={bk:>4d}, bmm={bmm:>2d}{marker}")
        else:
            print(f"  First 10 candidates (of {r['num_candidates']}):")
            for bm, bk, bmm in r['all_configs'][:10]:
                marker = " ★" if (bm, bk, bmm) == (r['default_BM'], r['default_BK'], r['default_bmm']) else ""
                print(f"    BM={bm:>4d}, BK={bk:>4d}, bmm={bmm:>2d}{marker}")


def generate_codegen_command(model_name, arch, results):
    """Generate the codegen command for a model."""
    BM_list = ",".join(str(r['default_BM']) for r in results)
    BK_list = ",".join(str(r['default_BK']) for r in results)
    bm_list = ",".join(str(r['default_bmm']) for r in results)

    script = f"codegen_{arch}.py"
    cmd = f'python utils/{script} --model {model_name} --BM {BM_list} --BK {BK_list} --bm {bm_list}'
    return cmd


def write_preset_config(model_name, arch, results, output_dir):
    """Write preset kernel configuration INI file."""
    config = ConfigParser()

    for i, r in enumerate(results):
        section = f'Kernels_{i}'
        config.add_section(section)
        config.set(section, 'm', str(r['M']))
        config.set(section, 'k', str(r['K']))
        config.set(section, 'bm', str(r['default_BM']))
        config.set(section, 'bk', str(r['default_BK']))
        config.set(section, 'bmm', str(r['default_bmm']))

    filename = f'kernel_config_{arch}.ini'
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'w') as f:
        config.write(f)

    return filepath


# =============================================================================
# Benchmarking
# =============================================================================

DEFAULT_BENCHMARK_SEQ_LEN = 128
DEFAULT_BENCHMARK_BATCH_SIZE = 1
DEFAULT_BENCHMARK_NUM_RUNS = 50


def benchmark_model_forward(model, tokenizer, sequence_length=DEFAULT_BENCHMARK_SEQ_LEN,
                            batch_size=DEFAULT_BENCHMARK_BATCH_SIZE,
                            num_runs=DEFAULT_BENCHMARK_NUM_RUNS):
    """
    Benchmark a LlamaForCausalLM model's forward pass.

    Measures average latency and tokens per second over *num_runs* forward
    passes using random input token ids.

    Parameters
    ----------
    model : torch.nn.Module
        The LLaMA model to benchmark.
    tokenizer : transformers.PreTrainedTokenizer or None
        If None, random token ids are generated directly.
    sequence_length : int
        Number of tokens in each input sequence.
    batch_size : int
        Number of sequences per batch.
    num_runs : int
        Number of forward passes to average over.

    Returns
    -------
    dict
        avg_latency_s, tokens_per_second
    """
    import torch

    device = "cpu"
    model = model.to(device).eval()
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    # Warm-up run
    with torch.no_grad():
        model(input_ids)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_ids)
            end = time.perf_counter()
            latencies.append(end - start)

    avg_latency = sum(latencies) / len(latencies)
    total_tokens = batch_size * sequence_length
    tokens_per_sec = total_tokens / avg_latency if avg_latency > 0 else 0.0

    return {
        "avg_latency_s": avg_latency,
        "tokens_per_second": tokens_per_sec,
    }


def benchmark_candidates(model_name, config_path, unique_shapes, arch,
                         sequence_length=DEFAULT_BENCHMARK_SEQ_LEN,
                         batch_size=DEFAULT_BENCHMARK_BATCH_SIZE,
                         num_runs=DEFAULT_BENCHMARK_NUM_RUNS):
    """
    Benchmark every candidate kernel configuration for a model.

    For each unique (M, K) shape the model is instantiated via
    ``LlamaForCausalLM.from_config()`` and benchmarked with a realistic
    forward-pass workload.  Each candidate is then logged with its
    parameters and the per-shape benchmark result.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g. 'Llama-80M').
    config_path : str
        Path to the model's config.json.
    unique_shapes : list[list[int]]
        Unique [[M, K], …] shapes extracted from the model.
    arch : str
        Architecture ('tl1' or 'tl2').
    sequence_length : int
        Sequence length for benchmark.
    batch_size : int
        Batch size for benchmark.
    num_runs : int
        Number of forward passes to average.

    Returns
    -------
    list[dict]
        One entry per (shape, candidate) combination.  Each dict contains:
        model_name, M, K, ROW_BLOCK_SIZE (BM), COL_BLOCK_SIZE (BK),
        PARALLEL_SIZE (bmm), avg_latency_s, tokens_per_second, rank.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    # Build the model once for benchmarking
    with open(config_path, "r") as fh:
        cfg_dict = json.load(fh)
    config = LlamaConfig(**cfg_dict)
    model = LlamaForCausalLM(config)

    # Run the model-level benchmark
    bench = benchmark_model_forward(
        model, tokenizer=None,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_runs=num_runs,
    )
    base_latency = bench["avg_latency_s"]
    base_tps = bench["tokens_per_second"]

    del model

    # Benchmark each candidate with a candidate-specific tiled microbenchmark.
    # The absolute throughput is anchored to the measured model forward pass,
    # while candidate rankings come from real blocked matmul timings.
    all_candidates = []
    for M, K in unique_shapes:
        configs = generate_configs_for_shape(M, K, arch)
        candidate_latencies = []
        for bm_val, bk_val, bmm_val in configs:
            candidate_latencies.append(
                _benchmark_candidate_shape_matmul(
                    M,
                    K,
                    bm_val,
                    bk_val,
                    bmm_val,
                    sequence_length,
                    num_runs,
                )
            )

        best_latency = min(lat["estimated_latency_s"] for lat in candidate_latencies)

        for (bm_val, bk_val, bmm_val), latency_info in zip(configs, candidate_latencies):
            relative_speedup = best_latency / latency_info["estimated_latency_s"]
            candidate_tps = base_tps * relative_speedup
            all_candidates.append({
                "model_name": model_name,
                "M": M,
                "K": K,
                "ROW_BLOCK_SIZE": bm_val,
                "COL_BLOCK_SIZE": bk_val,
                "PARALLEL_SIZE": bmm_val,
                "BM": bm_val,
                "BK": bk_val,
                "bm": bmm_val,
                "avg_latency_s": latency_info["estimated_latency_s"],
                "tile_latency_s": latency_info["tile_latency_s"],
                "tokens_per_second": candidate_tps,
            })

    # Rank candidates per shape by tokens_per_second (descending)
    shape_groups = defaultdict(list)
    for c in all_candidates:
        shape_groups[(c["M"], c["K"])].append(c)

    for key in shape_groups:
        group = sorted(shape_groups[key],
                       key=lambda x: x["tokens_per_second"], reverse=True)
        for rank, entry in enumerate(group, start=1):
            entry["rank"] = rank

    return all_candidates


def _benchmark_candidate_shape_matmul(M, K, BM, BK, bmm, sequence_length, num_runs):
    """
    Benchmark a single candidate configuration using a lightweight subtile matmul.

    Times a single (row_tile x BK) @ (BK x bmm) DGEMM and scales analytically
    by row_tiles * m_tiles * k_tiles * subtile_repeats to estimate total latency.
    This is much cheaper than the full-tile approach for large BM values.

    Returns
    -------
    dict
        estimated_latency_s for the full shape plus the measured subtile latency.
    """
    import torch

    row_tile = min(sequence_length, 32)
    x = torch.randn(row_tile, BK)
    w = torch.randn(BK, bmm)

    # Warm-up
    torch.mm(x, w)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        torch.mm(x, w)
        end = time.perf_counter()
        latencies.append(end - start)

    avg_subtile_latency = sum(latencies) / len(latencies)
    row_tiles = math.ceil(sequence_length / row_tile)
    m_tiles = math.ceil(M / BM)
    k_tiles = math.ceil(K / BK)
    subtile_repeats = math.ceil(BM / bmm)
    estimated_latency = avg_subtile_latency * row_tiles * m_tiles * k_tiles * subtile_repeats

    return {
        "tile_latency_s": avg_subtile_latency,
        "estimated_latency_s": estimated_latency,
    }


# =============================================================================
# Tuning Log I/O
# =============================================================================

def save_tuning_log(model_name, arch, candidates, log_dir):
    """
    Save the full tuning log for *model_name* / *arch* to a JSON file
    inside *log_dir*.

    Parameters
    ----------
    model_name : str
    arch : str
    candidates : list[dict]
        Output of :func:`benchmark_candidates`.
    log_dir : str
        Root tuning logs directory (e.g. ``tuning_logs``).

    Returns
    -------
    str
        Path to the written JSON file.
    """
    model_log_dir = os.path.join(log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    filepath = os.path.join(model_log_dir, f"tuning_log_{arch}.json")

    log_data = {
        "model_name": model_name,
        "arch": arch,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidates": candidates,
    }

    with open(filepath, "w") as fh:
        json.dump(log_data, fh, indent=2)

    return filepath


def load_tuning_log(filepath):
    """Load a previously saved tuning log JSON file."""
    with open(filepath, "r") as fh:
        return json.load(fh)


def select_best_from_benchmark(candidates):
    """
    Select the best kernel configuration for each unique (M, K) shape
    from benchmark results.

    For each shape the candidate with the highest ``tokens_per_second``
    (i.e. ``rank == 1``) is chosen.  The returned list uses the same
    ``default_BM`` / ``default_BK`` / ``default_bmm`` key names as
    :func:`generate_model_configs` so the result can be passed directly
    to :func:`write_preset_config` or ``save_tuned_config``.

    Parameters
    ----------
    candidates : list[dict]
        Output of :func:`benchmark_candidates`.

    Returns
    -------
    list[dict]
        One entry per unique (M, K) shape with keys:
        ``M``, ``K``, ``default_BM``, ``default_BK``, ``default_bmm``,
        ``num_candidates``, ``tokens_per_second``, ``avg_latency_s``.
        Shapes are returned in the order they first appear in *candidates*.
    """
    shape_groups = defaultdict(list)
    shape_order = []
    for c in candidates:
        key = (c["M"], c["K"])
        if key not in shape_groups:
            shape_order.append(key)
        shape_groups[key].append(c)

    results = []
    for key in shape_order:
        M, K = key
        group = shape_groups[key]
        best = max(group, key=lambda x: x["tokens_per_second"])
        results.append({
            "M": M,
            "K": K,
            "default_BM": best["BM"],
            "default_BK": best["BK"],
            "default_bmm": best["bm"],
            "num_candidates": len(group),
            "tokens_per_second": best["tokens_per_second"],
            "avg_latency_s": best["avg_latency_s"],
        })

    return results


def save_tuning_log_csv(model_name, arch, candidates, log_dir):
    """
    Save the tuning log for *model_name* / *arch* to a CSV summary file.

    The file is written as ``tuning_log_{arch}_summary_desc.csv`` inside
    ``{log_dir}/{model_name}/``, sorted in descending order of
    ``tokens_per_second`` so that the best candidates appear first.
    This format is compatible with :mod:`utils.generate_tuning_visualization`.

    Parameters
    ----------
    model_name : str
    arch : str
    candidates : list[dict]
        Output of :func:`benchmark_candidates`.
    log_dir : str
        Root tuning logs directory (e.g. ``tuning_logs``).

    Returns
    -------
    str
        Path to the written CSV file.
    """
    import csv

    model_log_dir = os.path.join(log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    filepath = os.path.join(model_log_dir, f"tuning_log_{arch}_summary_desc.csv")

    fieldnames = [
        "model_name", "M", "K",
        "ROW_BLOCK_SIZE", "COL_BLOCK_SIZE", "PARALLEL_SIZE",
        "BM", "BK", "bm",
        "avg_latency_s", "tile_latency_s", "tokens_per_second", "rank",
    ]

    sorted_candidates = sorted(
        candidates, key=lambda x: x["tokens_per_second"], reverse=True
    )

    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_candidates)

    return filepath


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BitNet Kernel Parameter Tuner - Generate optimal BM/BK/bm configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all models for ARM (TL1)
  python utils/kernel_tuner.py --all-models --arch tl1

  # Analyze specific model for x86 (TL2)
  python utils/kernel_tuner.py --model Llama-300M --arch tl2

  # Generate codegen commands
  python utils/kernel_tuner.py --all-models --arch tl1 --generate-commands

  # Write preset kernel configs
  python utils/kernel_tuner.py --all-models --arch tl1 --write-presets

  # Custom BM/BK search range
  python utils/kernel_tuner.py --model Llama-1B --arch tl1 \\
      --bm-range 128,256,512 --bk-range 64,128,256

  # Specify shapes directly (without model config)
  python utils/kernel_tuner.py --M 2048 --K 5632 --arch tl1
        """
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model', type=str, choices=list(MODEL_SHAPE_DICT.keys()),
                             help='Model name to tune')
    model_group.add_argument('--all-models', action='store_true',
                             help='Tune all supported models')
    model_group.add_argument('--new-models', action='store_true',
                             help='Tune only the 6 new model scales (80M-1B)')

    # Direct shape specification
    parser.add_argument('--M', type=int, help='Matrix M dimension (for single shape tuning)')
    parser.add_argument('--K', type=int, help='Matrix K dimension (for single shape tuning)')

    # Architecture
    parser.add_argument('--arch', type=str, required=True, choices=['tl1', 'tl2'],
                        help='Target architecture (tl1=ARM NEON, tl2=x86 AVX2)')

    # Custom parameter ranges
    parser.add_argument('--bm-range', type=str,
                        help='Comma-separated BM candidates (e.g., "128,256,512")')
    parser.add_argument('--bk-range', type=str,
                        help='Comma-separated BK candidates (e.g., "64,128,256")')

    # Output options
    parser.add_argument('--generate-commands', action='store_true',
                        help='Generate codegen shell commands')
    parser.add_argument('--write-presets', action='store_true',
                        help='Write preset kernel config INI files')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    # Parse custom ranges
    bm_range = [int(x) for x in args.bm_range.split(',')] if args.bm_range else None
    bk_range = [int(x) for x in args.bk_range.split(',')] if args.bk_range else None

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Handle direct shape specification
    if args.M is not None and args.K is not None:
        configs = generate_configs_for_shape(args.M, args.K, args.arch, bm_range, bk_range)
        default = select_default_config(args.M, args.K, args.arch, configs)

        print(f"\nShape: M={args.M}, K={args.K}, Arch={args.arch.upper()}")
        print(f"Valid configurations: {len(configs)}")
        if default:
            print(f"★ Recommended: BM={default[0]}, BK={default[1]}, bmm={default[2]}")
        print(f"\nAll candidates:")
        for bm, bk, bmm in configs:
            marker = " ★" if default and (bm, bk, bmm) == default else ""
            print(f"  BM={bm:>4d}, BK={bk:>4d}, bmm={bmm:>2d}{marker}")
        return

    # Determine which models to process
    new_model_names = ["Llama-80M", "Llama-120M", "Llama-300M",
                       "Llama-500M", "Llama-700M", "Llama-1B"]

    if args.all_models:
        models = list(MODEL_SHAPE_DICT.keys())
    elif args.new_models:
        models = new_model_names
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        sys.exit(1)

    all_json_results = {}

    for model_name in models:
        results = generate_model_configs(model_name, args.arch, bm_range, bk_range)

        if args.output_json:
            all_json_results[model_name] = [
                {
                    'M': r['M'], 'K': r['K'],
                    'default_BM': r['default_BM'],
                    'default_BK': r['default_BK'],
                    'default_bmm': r['default_bmm'],
                    'num_candidates': r['num_candidates'],
                }
                for r in results
            ]
        else:
            print_shape_analysis(model_name, args.arch, results)

        if args.generate_commands:
            # Only generate commands if all defaults are valid
            if all(r['default_BM'] is not None for r in results):
                cmd = generate_codegen_command(model_name, args.arch, results)
                print(f"\n  Codegen command:")
                print(f"    {cmd}")
            else:
                print(f"\n  ⚠ Cannot generate command: some shapes have no valid configuration")

        if args.write_presets:
            preset_dir = os.path.join(project_root, "preset_kernels", model_name)
            filepath = write_preset_config(model_name, args.arch, results, preset_dir)
            print(f"\n  ✅ Preset config written to: {filepath}")

    if args.output_json:
        print(json.dumps(all_json_results, indent=2))


if __name__ == "__main__":
    main()
