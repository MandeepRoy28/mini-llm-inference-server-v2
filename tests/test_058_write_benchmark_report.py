"""Tests for Problem 058 — Write a Benchmark Report."""

import importlib
import os
import tempfile
import pytest

try:
    mod = importlib.import_module("solutions.058_write_benchmark_report")
    write_benchmark_report = mod.write_benchmark_report
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 058 not found — skipping tests.", allow_module_level=True)

SAMPLE_RESULTS = {
    "model": "distilgpt2",
    "batch_size": 8,
    "page_size": 16,
    "throughput_req_per_sec": 14.2,
    "throughput_tokens_per_sec": 142.3,
    "ttft_p50": 0.043,
    "ttft_p95": 0.089,
    "ttft_p99": 0.121,
    "tpot_p50": 0.012,
    "tpot_p95": 0.019,
    "tpot_p99": 0.024,
    "gpu_memory_used_gb": 3.7,
}


def test_returns_string():
    """Function returns a non-empty string."""
    report = write_benchmark_report(SAMPLE_RESULTS)
    assert isinstance(report, str)
    assert len(report) > 0


def test_contains_model_name():
    """Report contains the model name from results."""
    report = write_benchmark_report(SAMPLE_RESULTS)
    assert "distilgpt2" in report


def test_contains_throughput():
    """Report contains the tokens-per-second throughput value."""
    report = write_benchmark_report(SAMPLE_RESULTS)
    assert "142.3" in report


def test_contains_separator():
    """Report contains a header/footer separator line."""
    report = write_benchmark_report(SAMPLE_RESULTS)
    assert "======" in report


def test_saves_to_file():
    """When output_path is given, report is written to that file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        path = f.name
    try:
        report = write_benchmark_report(SAMPLE_RESULTS, output_path=path)
        assert os.path.exists(path)
        with open(path) as fh:
            file_contents = fh.read()
        assert file_contents == report
    finally:
        os.unlink(path)
