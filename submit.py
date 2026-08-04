#!/usr/bin/env python3
"""
submit.py — Run tests for a problem and commit+push the solution to GitHub.

Usage:
    python submit.py <problem_id>        # e.g. python submit.py 001
    python submit.py 001                 # runs tests, then commits & pushes
    python submit.py 001 --test-only     # run tests without committing
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PROBLEMS_DIR = ROOT / "problems"
SOLUTIONS_DIR = ROOT / "solutions"
TESTS_DIR = ROOT / "tests"


def find_problem_file(problem_id: str) -> Path | None:
    padded = problem_id.zfill(3)
    matches = list(PROBLEMS_DIR.glob(f"{padded}_*.py"))
    return matches[0] if matches else None


def find_solution_file(problem_id: str) -> Path | None:
    padded = problem_id.zfill(3)
    matches = list(SOLUTIONS_DIR.glob(f"{padded}_*.py"))
    return matches[0] if matches else None


def find_test_file(problem_id: str) -> Path | None:
    padded = problem_id.zfill(3)
    matches = list(TESTS_DIR.glob(f"test_{padded}_*.py"))
    return matches[0] if matches else None


def run_tests(test_file: Path) -> bool:
    print(f"\n  Running tests: {test_file.name}")
    print("  " + "─" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
        cwd=ROOT,
    )
    return result.returncode == 0


def get_problem_name(problem_file: Path) -> str:
    m = re.match(r"^\d{3}_(.+)\.py$", problem_file.name)
    return m.group(1) if m else problem_file.stem


def commit_and_push(solution_file: Path, problem_id: str, problem_name: str) -> bool:
    name_display = problem_name.replace("_", " ")

    print(f"\n  Committing solution: {solution_file.name}")

    r = subprocess.run(["git", "add", str(solution_file)], cwd=ROOT)
    if r.returncode != 0:
        print("  ERROR: git add failed")
        return False

    commit_msg = f"solve {problem_id}: {name_display}"
    r = subprocess.run(["git", "commit", "-m", commit_msg], cwd=ROOT)
    if r.returncode != 0:
        print("  ERROR: git commit failed (nothing to commit?)")
        return False

    print(f"  Committed: \"{commit_msg}\"")

    r = subprocess.run(["git", "push"], cwd=ROOT)
    if r.returncode != 0:
        print("  ERROR: git push failed — check your remote is set up")
        return False

    print("  Pushed to GitHub ✓")
    return True


def update_progress_readme():
    """Regenerate the progress section in README.md after each submission."""
    from scaffold import get_all_problems, get_part_for, PARTS

    problems = get_all_problems()
    total = len(problems)
    solved = sum(1 for p in problems if p["solved"])

    lines = [
        "## Progress\n",
        f"**{solved}/{total} problems solved**\n\n",
        "| # | Problem | Status |\n",
        "|---|---------|--------|\n",
    ]

    current_part = None
    for p in problems:
        part_num, part_title = get_part_for(p["num"])
        if part_num != current_part:
            current_part = part_num
            lines.append(f"| | **Part {part_num} — {part_title}** | |\n")
        status = "✅" if p["solved"] else "⬜"
        name = p["name"].replace("_", " ")
        lines.append(f"| {p['num']:03d} | `{p['name']}` | {status} |\n")

    readme = ROOT / "README.md"
    if readme.exists():
        content = readme.read_text()
        # Replace everything between ## Progress and the next ## heading
        new_section = "".join(lines)
        content = re.sub(
            r"## Progress.*?(?=\n## |\Z)",
            new_section.rstrip(),
            content,
            flags=re.DOTALL,
        )
        readme.write_text(content)
    else:
        readme.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Submit a solution for a problem")
    parser.add_argument("problem_id", help="Problem number, e.g. 001 or 1")
    parser.add_argument("--test-only", action="store_true", help="Run tests without committing")
    args = parser.parse_args()

    problem_id = args.problem_id.zfill(3)

    # Check problem stub exists
    problem_file = find_problem_file(problem_id)
    if not problem_file:
        print(f"  ERROR: No problem file found for id {problem_id}")
        sys.exit(1)

    problem_name = get_problem_name(problem_file)
    print(f"\n  Problem {problem_id}: {problem_name.replace('_', ' ')}")

    # Check solution exists
    solution_file = find_solution_file(problem_id)
    if not solution_file:
        print(f"\n  No solution file found at solutions/{problem_id}_*.py")
        print(f"  Copy your solution from problems/{problem_file.name} to solutions/ and fill it in.")
        sys.exit(1)

    # Check test exists
    test_file = find_test_file(problem_id)
    if not test_file:
        print(f"  WARNING: No test file found for problem {problem_id}")
        print("  Skipping tests.")
        tests_passed = True
    else:
        tests_passed = run_tests(test_file)

    if not tests_passed:
        print(f"\n  ✗ Tests failed — fix your solution before submitting.")
        sys.exit(1)

    print(f"\n  ✓ All tests passed!")

    if args.test_only:
        print("  (--test-only mode: skipping commit)\n")
        sys.exit(0)

    # Update README progress table
    try:
        update_progress_readme()
        readme_path = ROOT / "README.md"
        subprocess.run(["git", "add", str(readme_path)], cwd=ROOT)
    except Exception as e:
        print(f"  WARNING: Could not update README: {e}")

    success = commit_and_push(solution_file, problem_id, problem_name)
    if success:
        print(f"\n  Solution {problem_id} locked in. Keep going!\n")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

