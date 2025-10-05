#!/usr/bin/env python3
"""
download_latest_fits.py
-----------------------
Download exactly ONE FITS file (the latest available for a single sector/quarter)
for a given target ID and mission (TESS or Kepler) using Lightkurve.

- Falls back gracefully if expected columns (e.g., 'sector') are absent by using alternatives
  or sorting by observation_time_start.
- For TESS, you can optionally choose a specific sector; otherwise the latest is used.
- For Kepler, you can optionally choose a specific quarter; otherwise the latest is used.
- You can also choose to fetch a TESSCut FFI cutout instead of a SPOC/QLP lightcurve.

Requirements:
    pip install lightkurve astroquery

Examples:
    python download_latest_fits.py --mission tess --id 25155310 --out ./data
    python download_latest_fits.py --mission tess --id 25155310 --sector 42 --prefer spoc --out ./data
    python download_latest_fits.py --mission tess --id 25155310 --tesscut --cutout-size 31 --out ./data
    python download_latest_fits.py --mission kepler --id 11446443 --out ./data
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import warnings
warnings.filterwarnings("ignore", message=".*tpfmodel submodule is not available.*", category=UserWarning)

import lightkurve as lk


def _target_name(mission: str, target_id: str) -> str:
    m = mission.strip().lower()
    if m == "tess":
        if not target_id.strip().lower().startswith("tic"):
            return f"TIC {target_id}"
        return target_id
    elif m == "kepler":
        if not target_id.strip().lower().startswith("kic"):
            return f"KIC {target_id}"
        return target_id
    else:
        raise ValueError("mission must be 'tess' or 'kepler'")


def _available_columns(sr: "lk.search.SearchResult") -> set:
    try:
        return set(sr.table.colnames)
    except Exception:
        return set()


def _pick_one_row_dynamic(
    sr: "lk.search.SearchResult",
    mission: str,
    value: Optional[int],
    author_priority: Sequence[str],
) -> "lk.search.SearchResult":
    """Pick exactly one row robustly.

    Strategy:
    - Determine grouping column candidates per mission:
        TESS: ['sector', 'sequence_number']
        Kepler: ['quarter', 'campaign']
    - If user specified value, filter on the first available matching column; if none, warn and ignore.
    - If no value, choose the max over the first available grouping column.
    - Apply author priority if available.
    - Sort by observation_time_start (or 'time' as fallback) and take the latest single row.
    """
    if len(sr) == 0:
        raise FileNotFoundError("No search results found.")

    mission = mission.lower().strip()
    cols = _available_columns(sr)
    if mission == "tess":
        group_cols = ["sector", "sequence_number"]
        value_label = "sector"
    elif mission == "kepler":
        group_cols = ["quarter", "campaign"]
        value_label = "quarter"
    else:
        group_cols = []
        value_label = "group"

    by_col = next((c for c in group_cols if c in cols), None)

    sr2 = sr
    chosen_value = None

    if value is not None:
        # Try to filter by the specified value on the first available group column
        for c in group_cols:
            if c in cols:
                try:
                    sr2 = sr[sr.table[c] == int(value)]
                    by_col = c
                    chosen_value = int(value)
                    break
                except Exception:
                    pass
        if len(sr2) == 0:
            raise FileNotFoundError(
                f"No results for requested {value_label}={value}. "
                f"Available columns: {sorted(cols)}"
            )
    elif by_col is not None:
        # Choose the latest by the available group column
        try:
            chosen_value = int(max(sr.table[by_col]))
            sr2 = sr[sr.table[by_col] == chosen_value]
        except Exception:
            # If anything goes wrong, fall back to temporal sorting below
            sr2 = sr

    # Apply author priority if present
    if "author" in _available_columns(sr2) and author_priority:
        for auth in author_priority:
            cand = sr2[sr2.table["author"] == auth]
            if len(cand) > 0:
                sr2 = cand
                break

    # Sort by observation start time and take the latest
    sort_col = "observation_time_start" if "observation_time_start" in _available_columns(sr2) else None
    if not sort_col and "time" in _available_columns(sr2):
        sort_col = "time"
    if sort_col:
        sr2 = sr2.sort(sort_col)

    if len(sr2) == 0:
        raise FileNotFoundError("No results left after filtering.")

    chosen = sr2[-1]  # single-row SearchResult

    # Debug log for transparency
    try:
        msg_bits = []
        if by_col is not None and chosen_value is not None:
            msg_bits.append(f"{by_col}={chosen_value}")
        if "author" in chosen.table.colnames:
            msg_bits.append(f"author={chosen.table['author'][0]}")
        if "productFilename" in chosen.table.colnames:
            msg_bits.append(f"file={chosen.table['productFilename'][0]}")
        print("Chosen product: " + "; ".join(msg_bits))
    except Exception:
        pass

    return chosen


def _find_actual_download(sink: Path, expected_basename: Optional[str]) -> Optional[Path]:
    if not sink.exists():
        return None
    fits_files = list(sink.rglob("*.fits"))
    if not fits_files:
        return None
    if expected_basename:
        for p in fits_files:
            if p.name == expected_basename:
                return p
    return max(fits_files, key=lambda p: p.stat().st_mtime)


def download_one_tess_lightcurve(
    target_id: str,
    sector: Optional[int],
    outdir: Path,
    author_priority: Sequence[str] = ("SPOC", "QLP"),
) -> Path:
    tname = _target_name("tess", target_id)
    sr = lk.search_lightcurve(tname, mission="TESS")
    row = _pick_one_row_dynamic(sr, mission="tess", value=sector, author_priority=author_priority)

    lc = row.download(download_dir=str(outdir))

    expected = None
    try:
        expected = lc.meta.get("FILENAME")
    except Exception:
        pass

    saved = _find_actual_download(outdir, expected_basename=expected)
    if not saved:
        raise RuntimeError("Download completed but could not locate the FITS file on disk.")
    return saved


def download_one_tesscut(
    target_id: str,
    sector: Optional[int],
    outdir: Path,
    cutout_size: int = 31,
) -> Path:
    tname = _target_name("tess", target_id)
    sr = lk.search_tesscut(tname)
    row = _pick_one_row_dynamic(sr, mission="tess", value=sector, author_priority=())

    tpf = row.download(cutout_size=cutout_size, download_dir=str(outdir))

    expected = None
    try:
        expected = tpf.hdu[0].header.get("FILENAME")
    except Exception:
        pass

    saved = _find_actual_download(outdir, expected_basename=expected)
    if not saved:
        raise RuntimeError("TESSCut download completed but could not locate FITS file on disk.")
    return saved


def download_one_kepler_lightcurve(
    target_id: str,
    quarter: Optional[int],
    outdir: Path,
) -> Path:
    tname = _target_name("kepler", target_id)
    sr = lk.search_lightcurve(tname, mission="Kepler")
    row = _pick_one_row_dynamic(sr, mission="kepler", value=quarter, author_priority=())

    lc = row.download(download_dir=str(outdir))

    expected = None
    try:
        expected = lc.meta.get("FILENAME")
    except Exception:
        pass

    saved = _find_actual_download(outdir, expected_basename=expected)
    if not saved:
        raise RuntimeError("Download completed but could not locate the FITS file on disk.")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download exactly ONE FITS for a target (TESS/Kepler).")
    parser.add_argument("--mission", required=True, choices=["tess", "kepler"], help="Mission name")
    parser.add_argument("--id", required=True, help="Target ID (TIC for TESS, KIC for Kepler; numeric OK)")

    # TESS controls
    parser.add_argument("--sector", type=int, default=None, help="TESS sector; if omitted, picks the latest")
    parser.add_argument("--tesscut", action="store_true", help="Use TESSCut FFI cutout instead of SPOC/QLP lightcurve")
    parser.add_argument("--cutout-size", type=int, default=31, help="TESSCut square size in pixels (default 31)")
    parser.add_argument("--prefer", choices=["spoc", "qlp", "any"], default="spoc",
                        help="For TESS lightcurves, prefer SPOC (default) or QLP products")

    # Kepler controls
    parser.add_argument("--quarter", type=int, default=None, help="Kepler quarter; if omitted, picks the latest")

    parser.add_argument("--out", default="./data", help="Download root directory")
    args = parser.parse_args()

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mission == "tess":
            if args.tesscut:
                saved = download_one_tesscut(args.id, args.sector, outdir, cutout_size=args.cutout_size)
            else:
                author_priority = {
                    "spoc": ("SPOC", "QLP"),
                    "qlp": ("QLP", "SPOC"),
                    "any": ()
                }[args.prefer]
                saved = download_one_tess_lightcurve(args.id, args.sector, outdir, author_priority=author_priority)
        else:  # kepler
            saved = download_one_kepler_lightcurve(args.id, args.quarter, outdir)

        print(str(saved))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
