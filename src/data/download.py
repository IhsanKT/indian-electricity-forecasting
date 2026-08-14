"""Download the Mendeley source workbooks.

Idempotent: a file already present at its manifest size is skipped, so re-running costs
nothing and a part-finished download is repaired rather than duplicated.

Mendeley sits behind Cloudflare, which rejects python-requests with HTTP 403 on TLS
fingerprint regardless of User-Agent; curl's fingerprint passes. File URLs answer 302,
hence -L. See docs/data_audit.md section 1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from src import config


def fetch_manifest(force: bool = False) -> list[dict[str, Any]]:
    """Return the dataset file manifest, caching it next to the raw files."""
    if config.MANIFEST_PATH.exists() and not force:
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    raw = _curl(["--max-time", "120", config.MENDELEY_MANIFEST_URL])
    manifest = json.loads(raw)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _curl(args: list[str]) -> bytes:
    proc = subprocess.run(["curl", "-sS", "--fail", *args], capture_output=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:400]}"
        )
    return proc.stdout


def download_all(verbose: bool = True) -> list[str]:
    """Download every workbook in the manifest. Returns the local filenames."""
    manifest = fetch_manifest()
    names: list[str] = []
    for entry in sorted(manifest, key=lambda e: e["filename"]):
        name = entry["filename"]
        dest = config.MENDELEY_RAW_DIR / name
        expected = entry["size"]
        names.append(name)
        if dest.exists() and dest.stat().st_size == expected:
            if verbose:
                print(f"  skip {name} (complete)", flush=True)
            continue
        if verbose:
            print(f"  downloading {name} ({expected/1e6:.1f} MB) ...", flush=True)
        _curl(["-L", "--max-time", "3600", "-o", str(dest),
               entry["content_details"]["download_url"]])
        got = dest.stat().st_size
        if got != expected:
            raise RuntimeError(f"{name}: downloaded {got} bytes, manifest says {expected}")
    return names


def main() -> int:
    print(f"Mendeley DOI {config.MENDELEY_DOI} -> {config.MENDELEY_RAW_DIR}")
    names = download_all()
    total = sum(p.stat().st_size for p in config.MENDELEY_RAW_DIR.glob("*.xlsx"))
    print(f"done: {len(names)} files, {total/1e6:.1f} MB on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
