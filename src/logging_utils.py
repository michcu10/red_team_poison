"""
logging_utils.py — Reusable run logger for ML experiment scripts.

Provides a Tee class and setup_run_logger context manager that duplicate
stdout/stderr to both the console and one or two log files simultaneously.
"""

import os
import sys
from contextlib import contextmanager
from datetime import datetime


class Tee:
    """Proxy for a stream that writes to the original stream and one or more files."""

    def __init__(self, original, *files):
        self._original = original
        self._files = list(files)

    # --- core write interface ---

    def write(self, data):
        self._original.write(data)
        for f in self._files:
            f.write(data)
        return len(data)

    def flush(self):
        self._original.flush()
        for f in self._files:
            f.flush()

    # --- attribute proxying so libraries that inspect sys.stdout don't break ---

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        # Return the underlying fd of the original stream; Tee itself has no fd.
        return self._original.fileno()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "replace")

    @property
    def closed(self):
        return False

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False

    def __getattr__(self, name):
        # Fall back to the original stream for any attribute not explicitly defined.
        return getattr(self._original, name)


@contextmanager
def setup_run_logger(script_name: str, output_dir: str = "results", log_path: str | None = None):
    """
    Context manager that redirects sys.stdout and sys.stderr to a Tee object
    writing to both the console and log files.

    Parameters
    ----------
    script_name : str
        Short identifier used as the filename prefix, e.g. "training" or "eval".
    output_dir : str
        Directory where log files are created (created automatically if absent).
    log_path : str | None
        Explicit path for the stable alias file.  When None the alias is
        auto-derived as ``<output_dir>/<script_name>_output.txt``.

    Yields
    ------
    timestamped_path : str
        Full path of the timestamped log file, e.g.
        ``results/training_20260421_145600.txt``.

    Usage
    -----
    >>> with setup_run_logger("training") as log_file:
    ...     print("This goes to console AND both log files")
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = os.path.join(output_dir, f"{script_name}_{timestamp}.txt")

    if log_path is None:
        log_path = os.path.join(output_dir, f"{script_name}_output.txt")

    timestamped_fh = open(timestamped_path, "w", encoding="utf-8", buffering=1)
    alias_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = Tee(original_stdout, timestamped_fh, alias_fh)
        sys.stderr = Tee(original_stderr, timestamped_fh, alias_fh)
        yield timestamped_path
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        timestamped_fh.close()
        alias_fh.close()
