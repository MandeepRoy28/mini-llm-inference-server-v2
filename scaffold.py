#!/usr/bin/env python3
"""
scaffold.py — Live progress tracker.
Shows which problems are solved (solutions/ exists) vs pending (problems/ only).
Run: python scaffold.py
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent
PROBLEMS_DIR = ROOT / "problems"
SOLUTIONS_DIR = ROOT / "solutions"

PARTS = {
    1: ("Tiny Transformer (Decoder-Only)", range(1, 10)),
    2: ("Sampling and Basic Generation", range(10, 17)),
    3: ("KV Cache", range(17, 25)),
    4: ("Paged Attention", range(25, 34)),
    5: ("Continuous Batching", range(34, 43)),
    6: ("Streaming Serving API", range(43, 51)),
    7: ("Throughput and Latency Benchmark Harness", range(51, 59)),
}


def get_all_problems() -> list[dict]:
    problems = []
    for path in sorted(PROBLEMS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        m = re.match(r"^(\d{3})_(.+)\.py$", path.name)
        if m:
            num = int(m.group(1))
            name = m.group(2)
            solution_path = SOLUTIONS_DIR / path.name
            problems.append({
                "num": num,
                "name": name,
                "filename": path.name,
                "solved": solution_path.exists(),
            })
    return problems


def get_part_for(num: int) -> tuple[int, str] | None:
    for part_num, (title, rng) in PARTS.items():
        if num in rng:
            return part_num, title
    return None, "Unknown"


def print_scaffold():
    problems = get_all_problems()
    total = len(problems)
    solved = sum(1 for p in problems if p["solved"])

    print()
    print(f"  Mini LLM Inference Server — Progress: {solved}/{total} solved")
    print("  " + "─" * 56)

    current_part = None
    for p in problems:
        part_num, part_title = get_part_for(p["num"])
        if part_num != current_part:
            current_part = part_num
            part_problems = [x for x in problems if get_part_for(x["num"])[0] == part_num]
            part_solved = sum(1 for x in part_problems if x["solved"])
            print()
            print(f"  PART {part_num} · {part_title.upper()}")
            print(f"  {part_solved}/{len(part_problems)} solved")
            print()

        status = "✓" if p["solved"] else "○"
        name_display = p["name"].replace("_", " ")
        print(f"    {status}  {p['num']:03d}  {name_display}")

    print()
    print("  " + "─" * 56)
    bar_filled = int((solved / total) * 40) if total else 0
    bar = "█" * bar_filled + "░" * (40 - bar_filled)
    pct = int((solved / total) * 100) if total else 0
    print(f"  [{bar}] {pct}%")
    print()


if __name__ == "__main__":
    print_scaffold()
