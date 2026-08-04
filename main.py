"""
main.py
=======
Boots the full Mini LLM Inference Server using your solutions.

Run:
    python3 main.py

As you solve more problems, more features unlock.

Usage:
    POST /generate          – full response (non-streaming)
    POST /generate/stream   – Server-Sent Events stream
    GET  /health            – liveness check
"""

from __future__ import annotations

import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# 1.  Track which solutions are present
# ---------------------------------------------------------------------------

missing: list[str] = []


def _missing(prob: str) -> None:
    missing.append(prob)


# ---------------------------------------------------------------------------
# 2.  Import engine singleton (problem 045)
# ---------------------------------------------------------------------------

try:
    from solutions.p045_build_inference_engine_singleton import (   # noqa: F401
        initialize_engine,
        get_inference_engine,
    )
    _HAS_ENGINE = True
except (ImportError, NotImplementedError, ModuleNotFoundError):
    try:
        # Try alternate naming convention (number prefix with underscore)
        import importlib
        mod = importlib.import_module(
            "solutions.045_build_inference_engine_singleton"
        )
        initialize_engine = mod.initialize_engine       # type: ignore[assignment]
        get_inference_engine = mod.get_inference_engine  # type: ignore[assignment]
        _HAS_ENGINE = True
    except Exception:
        _HAS_ENGINE = False
        _missing("045_build_inference_engine_singleton")

        def initialize_engine(*args, **kwargs):  # type: ignore[misc]
            raise RuntimeError(
                "Engine not available. Solve problem 045 first:\n"
                "  cp problems/045_build_inference_engine_singleton.py "
                "solutions/045_build_inference_engine_singleton.py"
            )

        def get_inference_engine():  # type: ignore[misc]
            raise RuntimeError(
                "Engine not available. Solve problem 045 first:\n"
                "  cp problems/045_build_inference_engine_singleton.py "
                "solutions/045_build_inference_engine_singleton.py"
            )


# ---------------------------------------------------------------------------
# 3.  Import SSE streaming app factory (problem 047)
# ---------------------------------------------------------------------------

def _load_streaming_app():
    """Try to load create_streaming_app from solutions."""
    try:
        import importlib
        mod = importlib.import_module(
            "solutions.047_build_sse_streaming_endpoint"
        )
        fn = mod.create_streaming_app
        return fn, True
    except Exception:
        return None, False


_create_streaming_app, _HAS_STREAMING = _load_streaming_app()
if not _HAS_STREAMING:
    _missing("047_build_sse_streaming_endpoint")


# ---------------------------------------------------------------------------
# 4.  Import non-streaming app factory (problem 048)
# ---------------------------------------------------------------------------

def _load_app():
    """Try to load create_app from solutions."""
    try:
        import importlib
        mod = importlib.import_module(
            "solutions.048_build_non_streaming_endpoint"
        )
        fn = mod.create_app
        return fn, True
    except Exception:
        return None, False


_create_app, _HAS_APP = _load_app()
if not _HAS_APP:
    _missing("048_build_non_streaming_endpoint")


# ---------------------------------------------------------------------------
# 5.  Print startup banner
# ---------------------------------------------------------------------------

PARTS = {
    "Part 1 – Tiny Transformer  (001-009)": list(range(1, 10)),
    "Part 2 – Sampling          (010-016)": list(range(10, 17)),
    "Part 3 – KV Cache          (017-024)": list(range(17, 25)),
    "Part 4 – Paged Attention   (025-033)": list(range(25, 34)),
    "Part 5 – Continuous Batching (034-042)": list(range(34, 43)),
    "Part 6 – Streaming API     (043-050)": list(range(43, 51)),
    "Part 7 – Benchmarks        (051-058)": list(range(51, 59)),
}


def _check_solutions() -> dict[str, bool]:
    """Return {problem_number_str: is_solved} for every problem."""
    import pathlib, importlib.util

    solutions_dir = pathlib.Path(__file__).parent / "solutions"
    solved: dict[str, bool] = {}
    for p in sorted(solutions_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        num = p.name[:3]
        spec = importlib.util.spec_from_file_location(p.stem, p)
        try:
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            solved[num] = True
        except (NotImplementedError, Exception):
            solved[num] = False
    return solved


def print_startup_banner() -> None:
    solved = _check_solutions()

    print()
    print("=" * 60)
    print("  Mini LLM Inference Server")
    print("=" * 60)
    print()
    all_done = 0
    all_total = 0
    for part_name, numbers in PARTS.items():
        done = sum(1 for n in numbers if solved.get(f"{n:03d}", False))
        total = len(numbers)
        all_done += done
        all_total += total
        status = "OK" if done == total else f"{done}/{total}"
        bar = "#" * done + "." * (total - done)
        print(f"  {status:>5}  [{bar}]  {part_name}")
    print()
    print(f"  Overall: {all_done}/{all_total} problems solved")
    print()

    if missing:
        print("  Missing solutions needed for the server:")
        for m in missing:
            print(f"    - {m}")
        print()
        print("  Endpoints will return 503 until these are solved.")
    else:
        print("  All required solutions present. Full server active.")
    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# 6.  Build the main FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mini LLM Inference Server",
    description=(
        "A from-scratch LLM inference server built by implementing 58 "
        "problems across 7 parts. Solve problems in solutions/ to unlock "
        "features."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Liveness probe."""
    solved_count = 58 - len(missing)
    return {
        "status": "ok",
        "engine_ready": _HAS_ENGINE,
        "streaming_ready": _HAS_STREAMING,
        "non_streaming_ready": _HAS_APP,
        "missing_solutions": missing,
        "message": f"{solved_count}/58 required components solved.",
    }


def _unsolved_response(endpoint: str, problems: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "not_implemented",
            "endpoint": endpoint,
            "message": (
                f"Endpoint '{endpoint}' requires unsolved problems. "
                f"Solve these first: {problems}"
            ),
            "hint": "cp problems/<name>.py solutions/<name>.py  then implement it.",
        },
    )


# ---------------------------------------------------------------------------
# 7.  Mount streaming app routes (if available)
# ---------------------------------------------------------------------------

if _HAS_STREAMING:
    try:
        _streaming_app = _create_streaming_app()
        app.mount("/stream", _streaming_app)

        # Also expose the canonical path directly
        from fastapi.routing import APIRoute  # noqa: E402

        for route in _streaming_app.routes:
            if isinstance(route, APIRoute):
                app.add_api_route(
                    route.path,
                    route.endpoint,
                    methods=list(route.methods or ["POST"]),
                    tags=["streaming"],
                )
    except Exception as exc:
        _HAS_STREAMING_MOUNTED = False

        @app.post("/generate/stream", tags=["streaming"])
        async def stream_placeholder():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "engine_error",
                    "detail": str(exc),
                    "message": "Streaming endpoint raised an error during setup.",
                },
            )
else:
    @app.post("/generate/stream", tags=["streaming"])
    async def stream_not_implemented():
        return _unsolved_response(
            "/generate/stream",
            ["047_build_sse_streaming_endpoint"],
        )


# ---------------------------------------------------------------------------
# 8.  Mount non-streaming app routes (if available)
# ---------------------------------------------------------------------------

if _HAS_APP:
    try:
        _main_app = _create_app()
        for route in _main_app.routes:
            if isinstance(route, APIRoute):
                app.add_api_route(
                    route.path,
                    route.endpoint,
                    methods=list(route.methods or ["POST"]),
                    tags=["generation"],
                )
    except Exception as exc:
        @app.post("/generate", tags=["generation"])
        async def generate_placeholder():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "engine_error",
                    "detail": str(exc),
                    "message": "Non-streaming endpoint raised an error during setup.",
                },
            )
else:
    @app.post("/generate", tags=["generation"])
    async def generate_not_implemented():
        return _unsolved_response(
            "/generate",
            ["048_build_non_streaming_endpoint"],
        )


# ---------------------------------------------------------------------------
# 9.  Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_startup_banner()

    port = 8000
    host = "0.0.0.0"

    print(f"Starting server on http://{host}:{port}")
    print("  Docs:    http://localhost:8000/docs")
    print("  Health:  http://localhost:8000/health")
    print()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
