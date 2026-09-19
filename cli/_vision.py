"""Shared access to the Vision helpers for the standalone scripts in tools/.

The landmark parsing lives in the `idphoto` package so there is exactly one
copy of it. It used to be duplicated here and a transposed min/max pair in the
copy silently put the eye boxes in the wrong place, which quietly moved the
under-eye brightening and the blemish keep-out. Importing beats re-deriving.

Landmarks come from Vision, not from numbers typed on the command line: the
coordinates are tied to the image's pixel grid, so hand-transcribed values go
stale the moment the input is resized or re-cropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root, so `import idphoto` finds the package beside cli/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from idphoto import matte, vision  # noqa: E402

__all__ = ["vision", "matte"]
