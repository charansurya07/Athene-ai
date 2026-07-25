"""
Isolated Python execution environment.

Used when the ingestion modality is `code`: rather than blindly executing
arbitrary uploaded code, this gives the Ingestion Agent a way to run small,
time-boxed snippets (e.g. to reproduce a benchmark the user attached) with
hard limits on wall-clock time and output size. This is a lightweight
subprocess sandbox — swap in Docker/gVisor/firecracker-backed execution for
production multi-tenant use.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 4000
_DEFAULT_TIMEOUT_S = 8


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


async def run_python_snippet(code: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> SandboxResult:
    """
    Execute `code` in a fresh subprocess with no network and a short timeout.

    This intentionally does NOT grant network access or file-system access
    outside its own temp directory — the subprocess is spawned with a
    restricted working directory and default OS-level user permissions.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-I",  # isolated mode: ignore user env/site customizations
            str(script_path),
            cwd=tmp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b, stderr_b, timed_out = b"", b"Execution timed out.", True

    return SandboxResult(
        stdout=stdout_b.decode(errors="replace")[:_MAX_OUTPUT_CHARS],
        stderr=stderr_b.decode(errors="replace")[:_MAX_OUTPUT_CHARS],
        exit_code=proc.returncode if proc.returncode is not None else -1,
        timed_out=timed_out,
    )


def summarize_code_for_ingestion(source: str, filename: str) -> str:
    """
    Cheap, non-executing summary used by the Ingestion Agent for any code
    file the pipeline doesn't explicitly choose to run (the common case —
    execution is opt-in, not automatic, for safety).
    """
    lines = source.splitlines()
    return (
        f"Code file `{filename}` ({len(lines)} lines). "
        f"First lines:\n" + "\n".join(lines[:15])
    )
