"""
PCAP Validator
==============
Validates uploaded PCAP files before any processing.
Checks: extension, file size, magic bytes, non-empty.

No external dependencies required — stdlib only.
"""
from pathlib import Path
from typing import Tuple

# ── PCAP magic numbers ────────────────────────────────────────────────────────
_PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1",   # PCAP little-endian
    b"\xa1\xb2\xc3\xd4",   # PCAP big-endian
    b"\x4d\x3c\xb2\xa1",   # PCAP nanosecond LE
    b"\xa1\xb2\x3c\x4d",   # PCAP nanosecond BE
    b"\x0a\x0d\x0d\x0a",   # PCAP-NG
}

VALID_EXTENSIONS = {".pcap", ".pcapng", ".cap"}
MAX_FILE_SIZE    = 100 * 1024 * 1024   # 100 MB


class PCAPValidationError(Exception):
    """Raised when PCAP file validation fails."""
    pass


def validate_pcap(file_path: Path) -> Tuple[bool, str]:
    """
    Validate a PCAP file.

    Returns:
        (True, description_message) on success.

    Raises:
        PCAPValidationError with a human-readable message on any failure.
    """
    # 1. Extension
    suffix = file_path.suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        raise PCAPValidationError(
            f"Invalid file extension '{suffix}'. "
            f"Accepted extensions: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    # 2. Existence
    if not file_path.exists():
        raise PCAPValidationError(f"File does not exist: {file_path.name}")

    # 3. Size
    size = file_path.stat().st_size
    if size == 0:
        raise PCAPValidationError("Uploaded file is empty (0 bytes).")
    if size > MAX_FILE_SIZE:
        mb = size / 1024 / 1024
        raise PCAPValidationError(
            f"File too large ({mb:.1f} MB). Maximum allowed: "
            f"{MAX_FILE_SIZE // 1024 // 1024} MB."
        )

    # 4. Minimum valid PCAP size (24-byte global header)
    if size < 24:
        raise PCAPValidationError(
            f"File too small ({size} bytes) to be a valid PCAP "
            f"(minimum 24 bytes for the global header)."
        )

    # 5. Magic bytes
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
    except PermissionError as exc:
        raise PCAPValidationError(f"Cannot read file (permission denied): {exc}")

    if magic not in _PCAP_MAGIC:
        raise PCAPValidationError(
            f"Not a valid PCAP file (bad magic bytes: {magic.hex()}). "
            "File may be corrupted or in an unsupported format."
        )

    kb = size / 1024
    return True, f"Valid PCAP file ({kb:.1f} KB)"
