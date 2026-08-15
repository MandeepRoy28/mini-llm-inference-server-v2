#!/usr/bin/env python3
"""
submit.py — Test a specific step, then commit and push model.py to GitHub.

Usage:
    python3 submit.py 004                  # test step 004, then commit + push
    python3 submit.py 004 --test-only      # just run the test, no commit
    python3 submit.py --all                # run all tests
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "model.py"
TEST_FILE = ROOT / "tests" / "test_model.py"


def run_test(step: str) -> bool:
    padded = step.zfill(3)
    test_name = f"test_{padded}_"
    print(f"\n  Running: pytest -k {test_name}")
    print("  " + "─" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-k", test_name, "-v", "--tb=short"],
        cwd=ROOT,
    )
    return result.returncode == 0


def run_all_tests() -> bool:
    print("\n  Running all tests...")
    print("  " + "─" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-v", "--tb=short"],
        cwd=ROOT,
    )
    return result.returncode == 0


def get_step_name(step: str) -> str:
    padded = step.zfill(3)
    import re
    if not MODEL_FILE.exists():
        return padded
    content = MODEL_FILE.read_text()
    m = re.search(rf"# Step {int(step)} - (\w+)", content)
    return m.group(1) if m else padded


def commit_and_push(step: str, step_name: str) -> bool:
    padded = step.zfill(3)
    print(f"\n  Committing model.py...")

    subprocess.run(["git", "add", "-A"], cwd=ROOT)

    commit_msg = f"solve {padded}: {step_name}"
    r = subprocess.run(["git", "commit", "-m", commit_msg], cwd=ROOT)
    if r.returncode != 0:
        print("  ERROR: git commit failed (nothing new to commit?)")
        return False

    print(f"  Committed: \"{commit_msg}\"")

    r = subprocess.run(["git", "push"], cwd=ROOT)
    if r.returncode != 0:
        print("  ERROR: git push failed")
        return False

    print("  Pushed to GitHub ✓")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", nargs="?", help="Step number, e.g. 004 or 4")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.all:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if not args.step:
        parser.print_help()
        sys.exit(1)

    step = args.step.lstrip("0") or "0"
    step_name = get_step_name(step)
    print(f"\n  Step {step.zfill(3)}: {step_name}")

    passed = run_test(step)
    if not passed:
        print(f"\n  ✗ Test failed — fix your implementation before submitting.")
        sys.exit(1)

    print(f"\n  ✓ Test passed!")

    if args.test_only:
        print("  (--test-only: skipping commit)\n")
        sys.exit(0)

    success = commit_and_push(step, step_name)
    if success:
        print(f"\n  Step {step.zfill(3)} locked in. Keep going!\n")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
