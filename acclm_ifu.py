import sys
from pathlib import Path
import os
ROOT_DIR = Path(__file__).parent.parent.absolute() # path to repo
sys.path.append(str(ROOT_DIR)) 
from functools import lru_cache
from utils.symbol import name_map
@lru_cache(maxsize=1)
def _acclm_library_files() -> tuple:
    """Return a cached tuple of PNG Paths in the ACCLM symbol library (empty if missing)."""

    # --- ACCLM symbol library (latest official symbol artwork) ---
    # Directory holding the reference PNGs each detected symbol is validated against.
    # The folder name starts with 'acclm_symbol_library' and carries a date suffix
    # (e.g. 'acclm_symbol_library_20260612_075855'); when several exist, the latest is
    # picked via descending name sort. It may be absent in some environments; lookups
    # degrade gracefully when missing.
    acclm_root = ROOT_DIR/ 'acclm'
    if not acclm_root.is_dir():
        return ()
    library_dirs = sorted(
        (d for d in acclm_root.glob('acclm_symbol_library*') if d.is_dir()),
        reverse=True,
    )
    if not library_dirs:
        return ()
    return tuple(library_dirs[0].glob('*.png'))


def find_acclm_template_path(class_name: str):
    """
    Locate the ACCLM library PNG for a detected symbol class.

    Lookup order:
        1. The class name returned by the model (e.g. 'RX_ONLY').
        2. The business name from name_map (e.g. 'RX Only Horizontal').
    ACCLM filenames may carry an irrelevant numeric prefix, e.g.
    '0028 - RX Only Horizontal.png'; that prefix is ignored when matching.

    Returns:
        Path of the matching file, or None when the symbol is not in the ACCLM
        library (in which case no ACCLM validation should be produced).
    """
    files = _acclm_library_files()
    if not files:
        return None
    candidates = [class_name]
    business_name = name_map.get(class_name)
    if business_name:  # skip empty business names
        candidates.append(business_name)
    for cand in candidates:
        for f in files:
            stem = f.stem  # filename without extension
            # return stem
            # exact match, or '<numeric prefix> - <name>' (prefix ignored)
            if stem == cand or stem.split(' - ', 1)[-1] == cand:
                return stem
    return None