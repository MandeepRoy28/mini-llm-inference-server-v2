"""Tests for Problem 055 — Plot Latency vs Batch Size."""

import importlib
import os
import tempfile
import pytest

try:
    mod = importlib.import_module("solutions.055_plot_latency_vs_batch_size")
    plot_latency_vs_batch_size = mod.plot_latency_vs_batch_size
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 055 not found — skipping tests.", allow_module_level=True)

SAMPLE_RESULTS = [
    {"batch_size": 1,  "ttft_mean": 0.03, "tpot_mean": 0.010},
    {"batch_size": 2,  "ttft_mean": 0.04, "tpot_mean": 0.011},
    {"batch_size": 4,  "ttft_mean": 0.05, "tpot_mean": 0.013},
    {"batch_size": 8,  "ttft_mean": 0.08, "tpot_mean": 0.017},
    {"batch_size": 16, "ttft_mean": 0.14, "tpot_mean": 0.025},
]


def test_saves_file():
    """Passing save_path creates a non-empty file on disk."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_latency_vs_batch_size(SAMPLE_RESULTS, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_returns_none():
    """Function returns None."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        result = plot_latency_vs_batch_size(SAMPLE_RESULTS, save_path=path)
        assert result is None
    finally:
        os.unlink(path)


def test_empty_results_no_raise():
    """Empty results list does not raise an exception."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_latency_vs_batch_size([], save_path=path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_single_point():
    """A single data point does not raise an exception."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_latency_vs_batch_size(
            [{"batch_size": 1, "ttft_mean": 0.03, "tpot_mean": 0.01}],
            save_path=path,
        )
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
