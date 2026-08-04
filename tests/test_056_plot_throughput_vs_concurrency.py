"""Tests for Problem 056 — Plot Throughput vs Concurrency."""

import importlib
import os
import tempfile
import pytest

try:
    mod = importlib.import_module("solutions.056_plot_throughput_vs_concurrency")
    plot_throughput_vs_concurrency = mod.plot_throughput_vs_concurrency
except (ModuleNotFoundError, AttributeError):
    pytest.skip("Solution 056 not found — skipping tests.", allow_module_level=True)

SAMPLE_RESULTS = [
    {"concurrency": 1,  "throughput_tokens_per_sec": 50},
    {"concurrency": 2,  "throughput_tokens_per_sec": 95},
    {"concurrency": 4,  "throughput_tokens_per_sec": 160},
    {"concurrency": 8,  "throughput_tokens_per_sec": 190},
    {"concurrency": 16, "throughput_tokens_per_sec": 195},
]


def test_saves_file():
    """Passing save_path creates a non-empty file on disk."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_throughput_vs_concurrency(SAMPLE_RESULTS, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_returns_none():
    """Function returns None."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        result = plot_throughput_vs_concurrency(SAMPLE_RESULTS, save_path=path)
        assert result is None
    finally:
        os.unlink(path)


def test_single_data_point_no_raise():
    """A single-element list does not raise an exception."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_throughput_vs_concurrency(
            [{"concurrency": 1, "throughput_tokens_per_sec": 50}],
            save_path=path,
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_monotone_input():
    """Strictly increasing throughput (no plateau) does not raise."""
    results = [
        {"concurrency": i, "throughput_tokens_per_sec": i * 20}
        for i in range(1, 6)
    ]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_throughput_vs_concurrency(results, save_path=path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
