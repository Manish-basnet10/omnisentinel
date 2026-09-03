"""
Zeek Service
============
Manages Zeek subprocess execution for PCAP analysis.

Responsibilities:
  - Discover Zeek executable safely (env var → PATH)
  - Create isolated temporary working directory
  - Run Zeek with JSON output and safe subprocess (no shell=True)
  - Parse conn.log → pandas DataFrame
  - Report Zeek version and log statistics
  - Clean up temp files on request

Raises ZeekNotFoundError / ZeekExecutionError — never crashes silently.
"""
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import pandas as pd

logger = logging.getLogger("omnisentinel.zeek")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "zeek_scripts"

ZEEK_TIMEOUT = 120   # seconds — hard cap on Zeek processing time


# ══════════════════════════════════════════════════════════════════════════════
# Custom Exceptions
# ══════════════════════════════════════════════════════════════════════════════
class ZeekNotFoundError(Exception):
    """Zeek executable not found or not executable."""


class ZeekExecutionError(Exception):
    """Zeek failed to process the PCAP or produced no usable output."""


# ══════════════════════════════════════════════════════════════════════════════
# Zeek discovery
# ══════════════════════════════════════════════════════════════════════════════
def find_zeek() -> Optional[str]:
    """
    Find the Zeek executable. Priority:
      1. ZEEK_PATH environment variable
      2. System PATH lookup
    Returns the absolute path string, or None if not found.
    """
    # 1. Environment override
    env_path = os.environ.get("ZEEK_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p)
        logger.warning(f"[ZEEK] ZEEK_PATH='{env_path}' is set but not executable — falling back to PATH.")

    # 2. PATH
    found = shutil.which("zeek")
    return found   # None if not on PATH


def get_zeek_version() -> Optional[str]:
    """Return Zeek version string (e.g. 'zeek version 8.2.2'), or None."""
    zeek_bin = find_zeek()
    if not zeek_bin:
        return None
    try:
        r = subprocess.run(
            [zeek_bin, "--version"],
            capture_output=True, text=True, timeout=10
        )
        return (r.stdout + r.stderr).strip()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Zeek execution
# ══════════════════════════════════════════════════════════════════════════════
def run_zeek(pcap_path: Path) -> Tuple[Path, dict]:
    """
    Run Zeek on *pcap_path*.

    Returns:
        (work_dir, stats_dict)
        work_dir — Path to the temp directory containing all Zeek output logs.
        **Caller is responsible for deleting work_dir when done.**

    Raises:
        ZeekNotFoundError  — if Zeek is not installed / on PATH.
        ZeekExecutionError — if Zeek fails or produces no usable conn.log.
    """
    zeek_bin = find_zeek()
    if not zeek_bin:
        raise ZeekNotFoundError(
            "Zeek executable not found.\n"
            "Install with:  brew install zeek\n"
            "Then verify:   zeek --version\n"
            "Or set:        export ZEEK_PATH=/path/to/zeek"
        )

    # Create an isolated temp directory for this run
    work_dir = Path(tempfile.mkdtemp(prefix="omnisentinel_zeek_"))

    logger.info(f"[ZEEK] Executable  : {zeek_bin}")
    logger.info(f"[ZEEK] Version     : {get_zeek_version()}")
    logger.info(
        f"[ZEEK] Input PCAP  : {pcap_path.name} "
        f"({pcap_path.stat().st_size / 1024:.1f} KB)"
    )
    logger.info(f"[ZEEK] Work dir    : {work_dir}")

    # ── Build command (no shell=True) ─────────────────────────────────────────
    cmd = [
        zeek_bin,
        "-r", str(pcap_path),               # read PCAP
        "LogAscii::use_json=T",             # JSON output
    ]

    # Load our custom cicflow policy if it exists
    cicflow = _SCRIPTS_DIR / "cicflow.zeek"
    if cicflow.exists():
        cmd.append(str(cicflow))
        logger.info(f"[ZEEK] Loading custom policy: cicflow.zeek")

    logger.info(f"[ZEEK] Running Zeek...")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ZEEK_TIMEOUT,
            cwd=str(work_dir),          # Zeek writes logs relative to CWD
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise ZeekExecutionError(
            f"Zeek processing timed out after {ZEEK_TIMEOUT}s. "
            "The PCAP may be too large. Try a smaller capture."
        )
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise ZeekExecutionError(f"Failed to execute Zeek: {exc}")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[ZEEK] Completed in {elapsed}s (exit code: {result.returncode})")

    if result.stderr:
        # Log Zeek stderr (often informational, not always fatal)
        for line in result.stderr.strip().splitlines()[:10]:
            logger.debug(f"[ZEEK stderr] {line}")

    # ── Validate output ───────────────────────────────────────────────────────
    conn_log = work_dir / "conn.log"
    log_files = sorted([f.name for f in work_dir.glob("*.log")])
    logger.info(f"[ZEEK] Log files produced: {log_files}")

    if not conn_log.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        err_snippet = result.stderr[:400] if result.stderr else "(no stderr)"
        raise ZeekExecutionError(
            "Zeek produced no conn.log. The PCAP may contain no complete flows, "
            f"or Zeek encountered an error.\nZeek stderr: {err_snippet}"
        )

    stats = {
        "zeek_binary": zeek_bin,
        "zeek_version": get_zeek_version(),
        "elapsed_seconds": elapsed,
        "return_code": result.returncode,
        "log_files": log_files,
    }
    return work_dir, stats


# ══════════════════════════════════════════════════════════════════════════════
# Log parsing
# ══════════════════════════════════════════════════════════════════════════════
def parse_conn_log(conn_log_path: Path) -> pd.DataFrame:
    """
    Parse Zeek conn.log (JSON format) into a pandas DataFrame.

    Zeek writes one JSON object per line when LogAscii::use_json=T.
    Skips comment lines (legacy TSV header lines starting with '#').
    """
    records: List[dict] = []

    with open(conn_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # silently skip malformed lines

    if not records:
        raise ZeekExecutionError(
            "conn.log exists but contains no parseable flow records. "
            "The PCAP may have captured only partial or non-IP traffic."
        )

    df = pd.DataFrame(records)
    logger.info(f"[ZEEK] Parsed {len(df):,} flows from conn.log")

    # Normalise 'duration' and numeric columns
    for col in ["duration", "orig_bytes", "resp_bytes", "orig_pkts",
                "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
