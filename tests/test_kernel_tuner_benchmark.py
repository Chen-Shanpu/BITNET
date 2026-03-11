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
    benchmark_candidates,
    benchmark_model_forward,
    generate_configs_for_shape,
    load_tuning_log,
    save_tuning_log,
    select_default_config,
)


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
