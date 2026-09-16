"""Diagnose saved grounding PNGs that render differently across viewers.

Symptom it diagnoses: a saved groundview.png / grounded.png shows the
full image in the browser (and inlined in vlm_report.html) but appears
TRUNCATED -- bottom half black -- in mspaint / PowerPoint.

Three checks, in order of likelihood:
  1. alpha channel: RGBA PNGs composite over BLACK in some MS tools but
     over WHITE in browsers -- a bottom-half-transparent image looks
     "missing" in paint and fine in Chrome. Re-saves a pure-RGB copy
     (*_fixed.png) that should display correctly everywhere.
  2. file truncation: PIL verify() catches a cut-off IDAT stream
     (interrupted copy, full disk).
  3. stale report: if the file bytes differ from the blobs inlined in
     vlm_report.html, a later run overwrote the PNGs and the report is
     from the older (good) run.

Usage (run inside the run directory):
    python check_png_integrity.py [names...]     # default: the two
                                                 # grounding PNGs + report
"""
import base64
import hashlib
import os
import re
import sys

DEFAULT_NAMES = ("groundview.png", "grounded.png")
_REPORT = "vlm_report.html"
_B64_RE = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")


def _report_blobs(run_dir: str) -> list[bytes]:
    path = os.path.join(run_dir, _REPORT)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [base64.b64decode(m) for m in _B64_RE.findall(f.read())]


def _check(name: str, run_dir: str, blobs: list[bytes]) -> None:
    from PIL import Image

    path = os.path.join(run_dir, name)
    print(f"== {name}")
    if not os.path.exists(path):
        print("  MISSING")
        return

    # 1) integrity: a truncated IDAT stream raises here
    try:
        Image.open(path).verify()
    except Exception as e:
        print(f"  TRUNCATED/CORRUPT file ({type(e).__name__}: {e}) "
              f"-> re-copy it from the machine that produced it")
        return

    im = Image.open(path)
    print(f"  mode={im.mode} size={im.size}")
    if im.mode in ("RGBA", "LA", "PA"):
        fixed = path[: -len(".png")] + "_fixed.png"
        im.convert("RGB").save(fixed)
        print(f"  ALPHA channel present (composites black in mspaint/"
              f"PPT, white in browsers) -> pure-RGB copy: {fixed}")
    else:
        print("  no alpha channel (RGB/gray)")

    # 2) stale-report check: same bytes as what vlm_report.html inlined?
    if blobs:
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        hit = any(hashlib.md5(b).hexdigest() == h for b in blobs)
        print(f"  identical to {_REPORT} inline copy: {hit}"
              + ("" if hit else
                 "  <- file was OVERWRITTEN after the report was built"
                 " (rerun in the same out_dir); the on-disk render is"
                 " the newer one"))
    else:
        print(f"  ({_REPORT} not found -- skipped the stale-report check)")


def main() -> None:
    run_dir = os.getcwd()
    names = sys.argv[1:] or list(DEFAULT_NAMES)
    blobs = _report_blobs(run_dir)
    for name in names:
        _check(name, run_dir, blobs)


if __name__ == "__main__":
    main()
