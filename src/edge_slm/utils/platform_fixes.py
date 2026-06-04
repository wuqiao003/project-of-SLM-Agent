"""
Windows platform fixes (UTF-8 for third-party packages that read UTF-8 files).
"""

import os
import sys


def ensure_utf8_windows() -> None:
    """
    Force UTF-8 before imports that read UTF-8 assets (e.g. trl *.jinja on Windows).

    Must run at process start, before importing trl/transformers training stacks.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
