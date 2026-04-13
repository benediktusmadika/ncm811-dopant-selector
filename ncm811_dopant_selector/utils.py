from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional

from .constants import (
    DEFAULT_SEED,
    HOST_SET,
    LOGGER_NAME,
    SUPPORTED_REASONING_EFFORTS,
    _ELEMENT_SYMBOLS,
    _ELEMENT_TOKEN_RE,
)
from .optional_deps import np

LOG = logging.getLogger(LOGGER_NAME)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def set_global_seed(seed: Optional[int]) -> Optional[int]:
    """
    Seed local stochastic sources used by this script.

    Note:
    - Python/NumPy can be seeded directly.
    - OpenAI Responses API generation is handled separately and may still vary
      across identical requests depending on model/server-side behavior.
    """
    if seed is None:
        return None

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    # Only affects child processes started after this point, but keeping it set
    # makes downstream tooling behavior more reproducible.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    txt = str(value).strip().lower()
    if not txt or txt == "auto":
        return None
    if txt not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {value}")
    return txt

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def clean_element_symbol(token: str) -> str:
    """
    Convert tokens like 'La3+' or 'Sr2+' or 'Al(III)' to a valid element symbol 'La','Sr','Al'.
    """
    if not token:
        return ""
    m = _ELEMENT_TOKEN_RE.search(token.strip())
    if not m:
        return ""
    sym = m.group(1)
    sym = sym[0].upper() + (sym[1:].lower() if len(sym) > 1 else "")
    return sym if sym in _ELEMENT_SYMBOLS else ""

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def parse_dopant_signature(signature: str) -> List[str]:
    """
    Parse a dopant signature such as "Al+La" into a normalized, sorted element list.
    Host elements are removed automatically.
    """
    if not signature:
        return []
    elements = []
    for token in re.split(r"[+,/;|]", str(signature)):
        symbol = clean_element_symbol(token)
        if symbol and symbol not in HOST_SET:
            elements.append(symbol)
    return sorted(dict.fromkeys(elements))
