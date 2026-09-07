#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from qera_original_a_isolation.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
