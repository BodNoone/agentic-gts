"""3DGS PLY I/O: parse full Gaussian attributes (means / scales / rots /
opacity / SH DC), with an in-process cache so repeated renders (god-view +
local evidence) only pay one disk read per run.

Only numpy is required here; rasterization lives in output/gs_render.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianData:
    means: np.ndarray       # (N,3) float32
    log_scales: np.ndarray  # (N,3) float32, log2-encoded scales
    quats: np.ndarray       # (N,4) float32, (w,x,y,z) unnormalized
    raw_opacity: np.ndarray  # (N,) float32, sigmoid-encoded
    f_dc: np.ndarray        # (N,3) float32, DC SH term

    def __len__(self):
        return len(self.means)


_CACHE: dict[str, GaussianData] = {}

_PROP_TYPES = {
    "float": ("f4", 4), "float32": ("f4", 4),
    "double": ("f8", 8), "float64": ("f8", 8),
    "int": ("i4", 4), "uint": ("u4", 4),
    "short": ("i2", 2), "ushort": ("u2", 2),
    "char": ("i1", 1), "uchar": ("u1", 1),
    "int8": ("i1", 1), "uint8": ("u1", 1),
}


def _parse_header(fp):
    """Return (fmt, elements) where elements = [(name, count, [(prop,type,name)])]."""
    if fp.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("PLY header truncated (no end_header)")
        parts = line.strip().split()
        if not parts:
            continue
        kw = parts[0].decode("ascii", "replace")
        if kw == "format":
            fmt = parts[1].decode()
        elif kw == "element":
            elements.append((parts[1].decode(), int(parts[2]), []))
        elif kw == "property":
            if parts[1] == b"list":
                # 3DGS exports never use list properties; refuse clearly
                raise ValueError("list property not supported for GS ply")
            elements[-1][2].append((parts[1].decode(), parts[2].decode()))
        elif kw == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing format line")
    return fmt, elements


def is_gaussian_ply(path: str) -> bool:
    """True if the PLY carries 3DGS attributes (f_dc / scale / rot / opacity)."""
    try:
        with open(path, "rb") as f:
            header = f.read(4096)
        head = header[:header.find(b"end_header") + 10]
        return b"f_dc_0" in head and b"opacity" in head
    except OSError:
        return False


def read_gaussian_ply(path: str, use_cache: bool = True) -> GaussianData:
    """Parse a 3DGS PLY into GaussianData. Cached by absolute path."""
    key = os.path.abspath(path)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    with open(path, "rb") as f:
        fmt, elements = _parse_header(f)
        # locate the vertex element
        vertex = next((e for e in elements if e[0] == "vertex"), None)
        if vertex is None:
            raise ValueError("PLY has no vertex element")
        _, count, props = vertex
        names = [p[1] for p in props]
        if "f_dc_0" not in names:
            raise ValueError("vertex lacks f_dc_0: not a 3DGS ply")
        if fmt == "binary_little_endian":
            dtype = np.dtype([(p[1], "<" + _PROP_TYPES[p[0]][0]) for p in props])
            arr = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
        elif fmt == "binary_big_endian":
            dtype = np.dtype([(p[1], ">" + _PROP_TYPES[p[0]][0]) for p in props])
            arr = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
        elif fmt == "ascii":
            rows = [f.readline().split() for _ in range(count)]
            cols = list(zip(*rows))
            arr = {n: np.array(c, dtype=np.float32) for n, c in zip(names, cols)}
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}")

    def col(*cands, fill=0.0):
        for c in cands:
            if c in names:
                return arr[c]
        v = np.full(count, fill, dtype=np.float32)
        return v

    means = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    log_scales = np.stack(
        [col("scale_0", fill=np.log(0.01)),
         col("scale_1", fill=np.log(0.01)),
         col("scale_2", fill=np.log(0.01))], axis=1).astype(np.float32)
    quats = np.stack(
        [col("rot_0", fill=1.0), col("rot_1"), col("rot_2"), col("rot_3")],
        axis=1).astype(np.float32)
    raw_opacity = col("opacity", fill=1.0).astype(np.float32)
    f_dc = np.stack(
        [col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1).astype(np.float32)
    gs = GaussianData(means=means, log_scales=log_scales, quats=quats,
                     raw_opacity=raw_opacity, f_dc=f_dc)
    if use_cache:
        _CACHE[key] = gs
    return gs


def write_gaussian_ply(path: str, gs: GaussianData) -> None:
    """Write a minimal 3DGS ply (used by tests)."""
    n = len(gs)
    props = [("float", "x"), ("float", "y"), ("float", "z"),
             ("float", "nx"), ("float", "ny"), ("float", "nz"),
             ("float", "f_dc_0"), ("float", "f_dc_1"), ("float", "f_dc_2"),
             ("float", "opacity"),
             ("float", "scale_0"), ("float", "scale_1"), ("float", "scale_2"),
             ("float", "rot_0"), ("float", "rot_1"),
             ("float", "rot_2"), ("float", "rot_3")]
    rows = np.zeros(n, dtype=np.dtype([(p[1], "<f4") for p in props]))
    for i, name in enumerate(["x", "y", "z"]):
        rows[name] = gs.means[:, i]
    for i, name in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        rows[name] = gs.f_dc[:, i]
    for i, name in enumerate(["scale_0", "scale_1", "scale_2"]):
        rows[name] = gs.log_scales[:, i]
    for i, name in enumerate(["rot_0", "rot_1", "rot_2", "rot_3"]):
        rows[name] = gs.quats[:, i]
    rows["opacity"] = gs.raw_opacity
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for t, name in props:
            f.write(f"property {t} {name}\n".encode())
        f.write(b"end_header\n")
        f.write(rows.tobytes())
