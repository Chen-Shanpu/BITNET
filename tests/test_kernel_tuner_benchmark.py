#!/usr/bin/env python3
"""
Tests for the kernel tuner benchmarking and logging functionality.

Run:
    python -m pytest tests/test_kernel_tuner_benchmark.py -v
"""

import json
import os
import sys

import pytest

# Ensure the project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from utils.kernel_tuner import (
    _benchmark_candidate_shape_matmul,
    benchmark_candidates,
    benchmark_model_forward,
    generate_configs_for_shape,
    load_tuning_log,
    save_tuning_log,
    save_tuning_log_csv,
    select_best_from_benchmark,
    select_default_config,
    write_preset_config,
)
import tune_all_models


# ---------------------------------------------------------------------------
# Tests for benchmark_model_forward
# ---------------------------------------------------------------------------

class TestBenchmarkModelForward:
    """Tests for the benchmark_model_forward() function."""

    def _make_model(self):
        """Create a minimal LLaMA model for testing."""
        from transformers import LlamaConfig, LlamaForCausalLM

        # Use a tiny model for fast tests
        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=256,
        )
        return LlamaForCausalLM(config)

    def test_returns_required_keys(self):
        model = self._make_model()
        result = benchmark_model_forward(model, tokenizer=None,
                                         sequence_length=8, batch_size=1,
                                         num_runs=2)
        assert "avg_latency_s" in result
        assert "tokens_per_second" in result

    def test_latency_is_positive(self):
        model = self._make_model()
        result = benchmark_model_forward(model, tokenizer=None,
                                         sequence_length=8, batch_size=1,
                                         num_runs=2)
        assert result["avg_latency_s"] > 0

    def test_tokens_per_second_is_positive(self):
        model = self._make_model()
        result = benchmark_model_forward(model, tokenizer=None,
                                         sequence_length=8, batch_size=1,
                                         num_runs=2)
        assert result["tokens_per_second"] > 0

    def test_custom_seq_len_and_batch(self):
        model = self._make_model()
        result = benchmark_model_forward(model, tokenizer=None,
                                         sequence_length=16, batch_size=2,
                                         num_runs=2)
        assert result["avg_latency_s"] > 0
        assert result["tokens_per_second"] > 0


# ---------------------------------------------------------------------------
# Tests for benchmark_candidates
# ---------------------------------------------------------------------------

class TestBenchmarkCandidates:
    """Tests for the benchmark_candidates() function."""

    @pytest.fixture
    def tiny_config_path(self, tmp_path):
        """Create a tiny model config.json for testing."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 256,
            "max_position_embeddings": 256,
        }
        path = tmp_path / "config.json"
        with open(path, "w") as f:
            json.dump(config, f)
        return str(path)

    def test_returns_list_of_dicts(self, tiny_config_path):
        shapes = [[64, 128], [128, 64]]
        candidates = benchmark_candidates(
            "test-model", tiny_config_path, shapes, "tl1",
            sequence_length=8, batch_size=1, num_runs=2,
        )
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert isinstance(candidates[0], dict)

    def test_candidate_has_required_fields(self, tiny_config_path):
        shapes = [[64, 128]]
        candidates = benchmark_candidates(
            "test-model", tiny_config_path, shapes, "tl1",
            sequence_length=8, batch_size=1, num_runs=2,
        )
        required_fields = [
            "model_name", "M", "K",
            "ROW_BLOCK_SIZE", "COL_BLOCK_SIZE", "PARALLEL_SIZE",
            "BM", "BK", "bm",
            "avg_latency_s", "tokens_per_second", "rank",
        ]
        for field in required_fields:
            assert field in candidates[0], f"Missing field: {field}"

    def test_ranks_are_assigned(self, tiny_config_path):
        shapes = [[64, 128]]
        candidates = benchmark_candidates(
            "test-model", tiny_config_path, shapes, "tl1",
            sequence_length=8, batch_size=1, num_runs=2,
        )
        ranks = [c["rank"] for c in candidates]
        assert min(ranks) >= 1
        assert max(ranks) <= len(candidates)

    def test_model_name_propagated(self, tiny_config_path):
        shapes = [[64, 128]]
        candidates = benchmark_candidates(
            "my-model", tiny_config_path, shapes, "tl2",
            sequence_length=8, batch_size=1, num_runs=2,
        )
        for c in candidates:
            assert c["model_name"] == "my-model"


# ---------------------------------------------------------------------------
# Tests for save_tuning_log / load_tuning_log
# ---------------------------------------------------------------------------

class TestTuningLogIO:
    """Tests for save_tuning_log() and load_tuning_log()."""

    def _sample_candidates(self):
        return [
            {
                "model_name": "Llama-80M",
                "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 128, "COL_BLOCK_SIZE": 64,
                "PARALLEL_SIZE": 64,
                "BM": 128, "BK": 64, "bm": 64,
                "avg_latency_s": 0.01,
                "tokens_per_second": 12800.0,
                "rank": 1,
            },
            {
                "model_name": "Llama-80M",
                "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 64, "COL_BLOCK_SIZE": 64,
                "PARALLEL_SIZE": 32,
                "BM": 64, "BK": 64, "bm": 32,
                "avg_latency_s": 0.01,
                "tokens_per_second": 12800.0,
                "rank": 2,
            },
        ]

    def test_save_creates_file(self, tmp_path):
        candidates = self._sample_candidates()
        path = save_tuning_log("Llama-80M", "tl1", candidates, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith("tuning_log_tl1.json")

    def test_save_creates_model_subdir(self, tmp_path):
        candidates = self._sample_candidates()
        save_tuning_log("Llama-80M", "tl1", candidates, str(tmp_path))
        assert os.path.isdir(os.path.join(str(tmp_path), "Llama-80M"))

    def test_round_trip(self, tmp_path):
        candidates = self._sample_candidates()
        path = save_tuning_log("Llama-80M", "tl1", candidates, str(tmp_path))
        loaded = load_tuning_log(path)

        assert loaded["model_name"] == "Llama-80M"
        assert loaded["arch"] == "tl1"
        assert len(loaded["candidates"]) == 2
        assert loaded["candidates"][0]["rank"] == 1
        assert loaded["candidates"][0]["tokens_per_second"] == 12800.0

    def test_log_contains_generated_at(self, tmp_path):
        candidates = self._sample_candidates()
        path = save_tuning_log("Llama-80M", "tl2", candidates, str(tmp_path))
        loaded = load_tuning_log(path)
        assert "generated_at" in loaded
        assert "UTC" in loaded["generated_at"]


# ---------------------------------------------------------------------------
# Tests for select_best_from_benchmark
# ---------------------------------------------------------------------------

class TestSelectBestFromBenchmark:
    """Tests for the select_best_from_benchmark() function."""

    def _sample_candidates(self):
        """Two shapes, two candidates each with different tokens_per_second."""
        return [
            # Shape (512, 512) – candidate A is better
            {
                "model_name": "Llama-80M", "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 128, "COL_BLOCK_SIZE": 64, "PARALLEL_SIZE": 64,
                "BM": 128, "BK": 64, "bm": 64,
                "avg_latency_s": 0.005, "tile_latency_s": 0.0001,
                "tokens_per_second": 20000.0, "rank": 1,
            },
            {
                "model_name": "Llama-80M", "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 64, "COL_BLOCK_SIZE": 64, "PARALLEL_SIZE": 32,
                "BM": 64, "BK": 64, "bm": 32,
                "avg_latency_s": 0.010, "tile_latency_s": 0.0002,
                "tokens_per_second": 10000.0, "rank": 2,
            },
            # Shape (1408, 512) – only one candidate
            {
                "model_name": "Llama-80M", "M": 1408, "K": 512,
                "ROW_BLOCK_SIZE": 128, "COL_BLOCK_SIZE": 64, "PARALLEL_SIZE": 64,
                "BM": 128, "BK": 64, "bm": 64,
                "avg_latency_s": 0.030, "tile_latency_s": 0.0002,
                "tokens_per_second": 5000.0, "rank": 1,
            },
        ]

    def test_returns_one_entry_per_shape(self):
        candidates = self._sample_candidates()
        results = select_best_from_benchmark(candidates)
        assert len(results) == 2

    def test_selects_highest_tokens_per_second(self):
        candidates = self._sample_candidates()
        results = select_best_from_benchmark(candidates)
        shape_512 = next(r for r in results if r["M"] == 512 and r["K"] == 512)
        assert shape_512["default_BM"] == 128
        assert shape_512["default_BK"] == 64
        assert shape_512["default_bmm"] == 64
        assert shape_512["tokens_per_second"] == 20000.0

    def test_returns_required_keys(self):
        candidates = self._sample_candidates()
        results = select_best_from_benchmark(candidates)
        required = {"M", "K", "default_BM", "default_BK", "default_bmm",
                    "num_candidates", "tokens_per_second", "avg_latency_s"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing keys in {r}"

    def test_num_candidates_is_shape_total(self):
        candidates = self._sample_candidates()
        results = select_best_from_benchmark(candidates)
        shape_512 = next(r for r in results if r["M"] == 512)
        assert shape_512["num_candidates"] == 2
        shape_1408 = next(r for r in results if r["M"] == 1408)
        assert shape_1408["num_candidates"] == 1

    def test_preserves_shape_discovery_order(self):
        candidates = self._sample_candidates()
        results = select_best_from_benchmark(candidates)
        assert results[0]["M"] == 512
        assert results[1]["M"] == 1408

    def test_empty_candidates(self):
        results = select_best_from_benchmark([])
        assert results == []


# ---------------------------------------------------------------------------
# Tests for save_tuning_log_csv
# ---------------------------------------------------------------------------

class TestSaveTuningLogCsv:
    """Tests for the save_tuning_log_csv() function."""

    def _sample_candidates(self):
        return [
            {
                "model_name": "Llama-80M", "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 128, "COL_BLOCK_SIZE": 64, "PARALLEL_SIZE": 64,
                "BM": 128, "BK": 64, "bm": 64,
                "avg_latency_s": 0.005, "tile_latency_s": 0.0001,
                "tokens_per_second": 20000.0, "rank": 1,
            },
            {
                "model_name": "Llama-80M", "M": 512, "K": 512,
                "ROW_BLOCK_SIZE": 64, "COL_BLOCK_SIZE": 64, "PARALLEL_SIZE": 32,
                "BM": 64, "BK": 64, "bm": 32,
                "avg_latency_s": 0.010, "tile_latency_s": 0.0002,
                "tokens_per_second": 10000.0, "rank": 2,
            },
        ]

    def test_creates_file(self, tmp_path):
        candidates = self._sample_candidates()
        path = save_tuning_log_csv("Llama-80M", "tl1", candidates, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith("tuning_log_tl1_summary_desc.csv")

    def test_creates_model_subdir(self, tmp_path):
        candidates = self._sample_candidates()
        save_tuning_log_csv("Llama-80M", "tl1", candidates, str(tmp_path))
        assert os.path.isdir(os.path.join(str(tmp_path), "Llama-80M"))

    def test_csv_has_header_and_rows(self, tmp_path):
        import csv
        candidates = self._sample_candidates()
        path = save_tuning_log_csv("Llama-80M", "tl2", candidates, str(tmp_path))
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert "tokens_per_second" in rows[0]
        assert "BM" in rows[0]

    def test_sorted_descending_by_tokens_per_second(self, tmp_path):
        import csv
        candidates = self._sample_candidates()
        path = save_tuning_log_csv("Llama-80M", "tl1", candidates, str(tmp_path))
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        tps_values = [float(r["tokens_per_second"]) for r in rows]
        assert tps_values == sorted(tps_values, reverse=True)

    def test_tl1_and_tl2_filenames_differ(self, tmp_path):
        candidates = self._sample_candidates()
        path1 = save_tuning_log_csv("Llama-80M", "tl1", candidates, str(tmp_path))
        path2 = save_tuning_log_csv("Llama-80M", "tl2", candidates, str(tmp_path))
        assert path1 != path2
        assert "tl1" in os.path.basename(path1)
        assert "tl2" in os.path.basename(path2)


# ---------------------------------------------------------------------------
# Tests for INI config completeness (all 6 tuning parameters)
# ---------------------------------------------------------------------------

_REQUIRED_INI_KEYS = {
    "row_block_size", "col_block_size", "parallel_size",  # canonical BitNet names
    "bm", "bk", "bmm",                                    # runtime shorthand aliases
}

_SAMPLE_RESULTS = [
    {"M": 512, "K": 512,  "default_BM": 128, "default_BK": 64, "default_bmm": 32},
    {"M": 512, "K": 1408, "default_BM": 128, "default_BK": 64, "default_bmm": 32},
]


class TestIniContainsAllSixParams:
    """
    Both write_preset_config() (kernel_tuner.py) and save_tuned_config()
    (tune_all_models.py) must write all 6 tuning parameters per section.
    """

    def _read_ini_keys(self, path):
        """Return a set of lowercase key names from the INI file."""
        from configparser import ConfigParser
        cfg = ConfigParser()
        cfg.read(path)
        keys = set()
        for section in cfg.sections():
            keys.update(k.lower() for k in cfg.options(section))
        return keys

    def test_write_preset_config_has_all_six_params(self, tmp_path):
        write_preset_config("Llama-test", "tl1", _SAMPLE_RESULTS, str(tmp_path))
        ini_path = str(tmp_path / "kernel_config_tl1.ini")
        keys = self._read_ini_keys(ini_path)
        assert _REQUIRED_INI_KEYS.issubset(keys), (
            f"Missing keys: {_REQUIRED_INI_KEYS - keys}"
        )

    def test_write_preset_config_tl2_has_all_six_params(self, tmp_path):
        write_preset_config("Llama-test", "tl2", _SAMPLE_RESULTS, str(tmp_path))
        ini_path = str(tmp_path / "kernel_config_tl2.ini")
        keys = self._read_ini_keys(ini_path)
        assert _REQUIRED_INI_KEYS.issubset(keys), (
            f"Missing keys: {_REQUIRED_INI_KEYS - keys}"
        )

    def test_save_tuned_config_has_all_six_params(self, tmp_path):
        tune_all_models.save_tuned_config(
            "Llama-test", "tl1", _SAMPLE_RESULTS, str(tmp_path)
        )
        ini_path = str(tmp_path / "Llama-test" / "kernel_config_tl1.ini")
        keys = self._read_ini_keys(ini_path)
        assert _REQUIRED_INI_KEYS.issubset(keys), (
            f"Missing keys: {_REQUIRED_INI_KEYS - keys}"
        )

    def test_ini_values_match_result_dict(self, tmp_path):
        """ROW_BLOCK_SIZE == BM == bm, etc. for every section."""
        from configparser import ConfigParser
        write_preset_config("Llama-test", "tl1", _SAMPLE_RESULTS, str(tmp_path))
        cfg = ConfigParser()
        cfg.read(str(tmp_path / "kernel_config_tl1.ini"))
        for i, r in enumerate(_SAMPLE_RESULTS):
            sec = f"Kernels_{i}"
            assert cfg.getint(sec, "ROW_BLOCK_SIZE") == r["default_BM"]
            assert cfg.getint(sec, "COL_BLOCK_SIZE") == r["default_BK"]
            assert cfg.getint(sec, "PARALLEL_SIZE")  == r["default_bmm"]
            assert cfg.getint(sec, "bm")  == r["default_BM"]
            assert cfg.getint(sec, "bk")  == r["default_BK"]
            assert cfg.getint(sec, "bmm") == r["default_bmm"]

    def test_ini_has_six_not_three_param_keys(self, tmp_path):
        """Regression: the old format only had bm/bk/bmm; new format must have 6."""
        write_preset_config("Llama-test", "tl1", _SAMPLE_RESULTS[:1], str(tmp_path))
        from configparser import ConfigParser
        cfg = ConfigParser()
        cfg.read(str(tmp_path / "kernel_config_tl1.ini"))
        sec = "Kernels_0"
        param_keys = {k for k in cfg.options(sec) if k not in ("m", "k")}
        assert len(param_keys) == 6, (
            f"Expected 6 tuning-param keys, got {len(param_keys)}: {param_keys}"
        )


# ---------------------------------------------------------------------------
# Tests for BitNet quantized microbenchmark
# ---------------------------------------------------------------------------

class TestBitNetQuantizedMicrobenchmark:
    """
    _benchmark_candidate_shape_matmul must use ternary int8 weights and
    int8 activations, not FP32 random tensors.
    """

    def test_returns_required_keys(self):
        result = _benchmark_candidate_shape_matmul(512, 512, 128, 64, 32, 32, 5)
        assert "tile_latency_s" in result
        assert "estimated_latency_s" in result

    def test_latencies_are_positive(self):
        result = _benchmark_candidate_shape_matmul(512, 512, 128, 64, 32, 32, 5)
        assert result["tile_latency_s"] > 0
        assert result["estimated_latency_s"] > 0

    def test_estimated_exceeds_tile_latency(self):
        """Full-shape estimate must be >= single-tile time."""
        result = _benchmark_candidate_shape_matmul(1024, 512, 128, 64, 32, 32, 5)
        assert result["estimated_latency_s"] >= result["tile_latency_s"]

    def test_uses_integer_valued_inputs(self):
        """
        Verify that the microbenchmark function builds integer-valued tensors
        (ternary weights, int8 activations) by inspecting the source code.
        The previous FP32 implementation used torch.randn; the BitNet
        implementation must use torch.randint.
        """
        import inspect
        src = inspect.getsource(_benchmark_candidate_shape_matmul)
        # Must not use torch.randn (FP32) for weight/activation creation
        assert "torch.randn" not in src, (
            "_benchmark_candidate_shape_matmul still uses torch.randn; "
            "it must use torch.randint for BitNet ternary int8 ops."
        )
        # Must use torch.randint (integer inputs)
        assert "torch.randint" in src, (
            "_benchmark_candidate_shape_matmul must use torch.randint "
            "for int8/ternary tensor creation."
        )

    def test_smaller_bk_gives_shorter_tile_latency(self):
        """Halving BK should reduce tile latency (smaller matmul)."""
        r_large = _benchmark_candidate_shape_matmul(512, 512, 128, 64, 32, 32, 10)
        r_small = _benchmark_candidate_shape_matmul(512, 512, 128, 32, 32, 32, 10)
        # Smaller BK → smaller tile → should be faster or equal
        # Allow 3× tolerance for CPU timing noise
        assert r_small["tile_latency_s"] <= r_large["tile_latency_s"] * 3
