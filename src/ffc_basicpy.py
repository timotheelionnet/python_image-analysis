# This ffc_basicpy.py file contains a series of functions 
# that will be usd to perform flat-field correction in the BaSiC_FFC notebook

from __future__ import annotations
from pathlib import Path
import numpy as np
from tifffile import imwrite
from basicpy import BaSiC

# ----------------------------
# Shape & I/O helpers
# ----------------------------

def to_cyx(arr, *, channels_last: bool = False, swap_xy: bool = False) -> np.ndarray:
    """
    Ensure array is [C, Y, X]. Accepts [Y,X], [C,Y,X], or [Y,X,C] (if channels_last=True).
    Optionally swap X<->Y after conversion (swap_xy=True) if your data is actually [C,X,Y].
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        cyx = a[np.newaxis, ...]
    elif a.ndim == 3:
        cyx = np.moveaxis(a, -1, 0) if channels_last else a
    else:
        raise ValueError(f"Unsupported shape {a.shape}")
    if swap_xy:
        cyx = np.swapaxes(cyx, 1, 2)
    return cyx

def write_image_cyx(path: Path | str, arr_cyx: np.ndarray, *, ome: bool = True) -> None:
    """
    Write [C,Y,X] to disk. If C==1, writes [Y,X]. Adds OME axes metadata.
    """
    arr = np.asarray(arr_cyx)
    if arr.ndim != 3:
        raise ValueError(f"Expected [C,Y,X], got {arr.shape}")
    if arr.shape[0] == 1:
        imwrite(str(path), arr[0], photometric="minisblack", ome=ome, metadata={"axes": "YX"})
    else:
        imwrite(str(path), arr, photometric="minisblack", ome=ome, metadata={"axes": "CYX"})

# ----------------------------
# BaSiC estimation & apply
# ----------------------------

def _stack_for_basic(planes_list, sample_fraction: float, random_state: int) -> np.ndarray:
    """
    planes_list: list of [Y,X]. Return float32 [N,Y,X], optionally subsampled.
    """
    stack = np.stack(planes_list, axis=0).astype(np.float32, copy=False)
    if sample_fraction >= 1.0 or stack.shape[0] <= 2:
        return stack
    rng = np.random.default_rng(random_state)
    k = max(2, int(np.ceil(sample_fraction * stack.shape[0])))
    idx = rng.choice(stack.shape[0], size=k, replace=False)
    return stack[idx]

def estimate_basic_flat_dark(
    planes_list,
    *,
    get_darkfield: bool = False,
    sample_fraction: float = 1.0,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Estimate flatfield (and optional darkfield) for one group×channel from multiple positions.
    """
    stack = _stack_for_basic(planes_list, sample_fraction, random_state)
    basic = BaSiC(get_darkfield=get_darkfield)  # keep constructor minimal (newer BaSiCPy forbids extra kwargs)
    basic.fit(stack)
    flat = np.maximum(basic.flatfield.astype(np.float32), 1e-6)
    dark = basic.darkfield.astype(np.float32) if get_darkfield else None
    return flat, dark

def apply_basic_per_channel(
    img_cyx: np.ndarray,
    flats: list[np.ndarray],
    darks: list[np.ndarray | None] | None,
    *,
    preserve_mean: bool = True,
    out_dtype=None,
    clip: bool = True,
) -> np.ndarray:
    """
    Apply per-channel correction to [C,Y,X] using lists of flat/dark maps aligned to channels.
    """
    C, Y, X = img_cyx.shape
    out = np.empty_like(img_cyx, dtype=np.float32)
    eps = 1e-6
    for c in range(C):
        flat = flats[c].astype(np.float32, copy=False)
        dark = None if (darks is None or darks[c] is None) else darks[c].astype(np.float32, copy=False)
        imgf = img_cyx[c].astype(np.float32, copy=False)
        if dark is not None:
            imgf = np.maximum(imgf - dark, 0.0)
        corr = imgf / (flat + eps)
        if preserve_mean:
            corr *= float(np.mean(flat))
        out[c] = corr

    if out_dtype is None:
        out_dtype = img_cyx.dtype if np.issubdtype(img_cyx.dtype, np.integer) else np.float32

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        if clip:
            out = np.clip(out, info.min, info.max)
    return out.astype(out_dtype, copy=False)

# ----------------------------
# Cache helpers (generic; notebook supplies base_dir)
# ----------------------------

def cache_key(base_dir: Path, group_name: str, ch: int) -> Path:
    safe_group = str(group_name).replace("/", "_").replace("\\", "_")
    return Path(base_dir) / f"basic_map__group={safe_group}__ch={ch}.npz"

def have_cached(base_dir: Path, group_name: str, ch: int) -> bool:
    return cache_key(base_dir, group_name, ch).exists()

def save_cache(base_dir: Path, group_name: str, ch: int, flat: np.ndarray, dark: np.ndarray | None) -> None:
    np.savez_compressed(
        cache_key(base_dir, group_name, ch),
        flat=flat.astype(np.float32),
        dark=(np.array([]) if dark is None else dark.astype(np.float32)),
    )

def load_cache(base_dir: Path, group_name: str, ch: int) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(cache_key(base_dir, group_name, ch))
    flat = data["flat"].astype(np.float32)
    dark = data["dark"].astype(np.float32)
    if dark.size == 0:
        dark = None
    return flat, dark
