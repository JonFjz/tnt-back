
"""
Download exactly ONE FITS (latest single sector/quarter) for a star from MAST via REST.

- Missions: TESS or Kepler
- Input ID may be numeric or prefixed (e.g., "25155310" or "TIC 25155310"; "11446443" or "KIC 11446443")
- Optionally pin a sector (--sector) or quarter (--quarter); otherwise the latest is auto-selected
- Chooses a single LIGHTCURVE file (SPOC/QLP/Kepler LLC/SLC) and downloads it

Usage:
  pip install requests
  python download_fits_mast.py --mission tess   --id 25155310 --out ./data
  python download_fits_mast.py --mission kepler --id 11446443 --out ./data
  python download_fits_mast.py --mission tess   --id 25155310 --sector 42 --out ./data
  # prefer "tpf" if you really want a target pixel file instead of a light curve:
  python download_fits_mast.py --mission tess --id 25155310 --prefer tpf

"""

import argparse
import os
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
from urllib.parse import quote as urlencode

# API "invoke" wrapper
def mast_query(request: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # https://mast.stsci.edu/api/v0/pyex.html
    request_url = 'https://mast.stsci.edu/api/v0/invoke'
    version = ".".join(map(str, sys.version_info[:3]))
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent": "python-requests/" + version,
    }
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    resp = requests.post(request_url, data="request=" + req_string, headers=headers)
    resp.raise_for_status()
    head = dict(resp.headers)
    content = resp.content.decode("utf-8")
    return head, json.loads(content)

# Direct download endpoint
def download_request(payload, filename: str, download_type: str = "file") -> str:
    # https://mast.stsci.edu/api/v0/pyex.html
    request_url = 'https://mast.stsci.edu/api/v0.1/Download/' + download_type
    r = requests.post(request_url, data=payload)
    r.raise_for_status()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(r.content)
    return filename

# ---------- Utilities ----------
def normalize_target(mission: str, target_id: str) -> str:
    t = target_id.strip()
    m = mission.strip().lower()
    if m == "tess":
        return t if t.lower().startswith("tic") else f"TIC {t}"
    elif m == "kepler":
        return t if t.lower().startswith("kic") else f"KIC {t}"
    else:
        raise ValueError("mission must be 'tess' or 'kepler'")

def resolve_name(name: str) -> Tuple[float, float]:
    # Mast.Name.Lookup
    req = {
        "service": "Mast.Name.Lookup",
        "params": {"input": name, "format": "json"},
    }
    _, out = mast_query(req)  # returns dict
    data = out.get("resolvedCoordinate")
    if not data:
        # Some versions return under ['data'][0]
        dat = out.get("data")
        if dat and isinstance(dat, list) and dat:
            ra = float(dat[0]["ra"])
            dec = float(dat[0]["decl"])
            return ra, dec
        raise FileNotFoundError(f"Name resolver failed for '{name}'.")
    return float(data["ra"]), float(data["decl"])

def caom_cone(ra: float, dec: float, radius_deg: float = 0.003) -> Dict[str, Any]:
    # Mast.Caom.Cone (small radius around the star)
    cachebreaker = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    req = {
        "service": "Mast.Caom.Cone",
        "params": {
            "ra": ra,
            "dec": dec,
            "radius": radius_deg,  # degrees
            "cachebreaker": cachebreaker
        },
        "format": "json",
        "pagesize": 5000,
        "page": 1,
        "removenullcolumns": True,
        "timeout": 5,
        "cachebreaker": cachebreaker
    }
    # Sometimes returns status EXECUTING briefly
    while True:
        _, out = mast_query(req)
        status = out.get("status", "COMPLETE")
        if status != "EXECUTING":
            return out
        time.sleep(0.5)

def filter_obs(rows: List[Dict[str, Any]], mission: str) -> List[Dict[str, Any]]:
    m = mission.lower()
    keep = []
    for r in rows:
        ocoll = (r.get("obs_collection") or "").lower()
        dpt   = (r.get("dataproduct_type") or "").lower()
        # Mission match
        if m == "tess" and ocoll not in ("tess", "hlsp"):
            continue
        if m == "kepler" and ocoll not in ("kepler", "hlsp"):
            continue
        # We care about time series for LC/TPF selections
        if dpt != "timeseries":
            continue
        keep.append(r)
    return keep

def choose_obs(rows: List[Dict[str, Any]], wanted_seq: Optional[int]) -> Dict[str, Any]:
    if not rows:
        raise FileNotFoundError("No CAOM rows after filtering.")
    # Use CAOM sequence_number (Kepler quarter or TESS sector)
    # pick desired or max available
    seqs = [r.get("sequence_number") for r in rows if r.get("sequence_number") is not None]
    if not seqs:
        # Fallback: sort by t_min/t_max and pick latest
        rows_sorted = sorted(rows, key=lambda r: (r.get("t_max") or 0.0, r.get("t_min") or 0.0))
        return rows_sorted[-1]
    if wanted_seq is None:
        target_seq = max(seqs)
    else:
        target_seq = int(wanted_seq)
    cand = [r for r in rows if r.get("sequence_number") == target_seq]
    if not cand:
        raise FileNotFoundError(f"No observation for sequence_number={target_seq}")
    # If multiple, pick the one with latest t_max
    cand.sort(key=lambda r: (r.get("t_max") or 0.0, r.get("t_min") or 0.0))
    return cand[-1]

def caom_products(obsid: int) -> Dict[str, Any]:
    req = {
        "service": "Mast.Caom.Products",
        "params": {"obsid": obsid},
        "format": "json",
        "pagesize": 200,
        "page": 1
    }
    _, out = mast_query(req)
    return out

def pick_one_product(products: List[Dict[str, Any]], prefer: str) -> Dict[str, Any]:
    """
    Prefer LC FITS: TESS SPOC *_lc.fits, TESS QLP *_llc.fits, Kepler *_llc.fits > *_slc.fits
    If none, optionally TPF, else any .fits.
    """
    fits = [p for p in products if str(p.get("productFilename", "")).lower().endswith(".fits")]
    if not fits:
        raise FileNotFoundError("No FITS products found under this CAOM obsid.")

    prefer = prefer.lower().strip()
    name = lambda p: p.get("productFilename", "").lower()
    subgroup = lambda p: (p.get("productSubGroupDescription") or "").lower()

    # Rank list: tuples (rank, product)
    ranked = []
    for p in fits:
        fn = name(p)
        sub = subgroup(p)
        # High priority: explicit lightcurve subgroup
        r = 100
        if "lightcurve" in sub:
            r -= 50
        # Filename heuristics
        if fn.endswith("_lc.fits"):
            r -= 40
        if fn.endswith("_llc.fits"):
            r -= 38
        if fn.endswith("_slc.fits"):
            r -= 36
        if "_lc.fits" in fn or "_llc.fits" in fn or "_slc.fits" in fn:
            r -= 10

        # Prefer switch (lc vs tpf)
        is_tpf = ("tpf" in fn) or ("targetpixel" in sub)
        if prefer == "lc" and is_tpf:
            r += 20
        if prefer == "tpf" and not is_tpf:
            r += 20

        # Small nudge for SPOC vs QLP if present in filename
        if "spoc" in fn:
            r -= 3
        if "qlp" in fn:
            r -= 1

        ranked.append((r, p))

    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]

def download_one(mission: str, target_id: str,
                 sector: Optional[int],
                 quarter: Optional[int],
                 outdir: Path,
                 prefer: str = "lc") -> Path:
    target = normalize_target(mission, target_id)
    ra, dec = resolve_name(target)  # Mast.Name.Lookup
    cone = caom_cone(ra, dec, radius_deg=0.003)  # Mast.Caom.Cone
    rows = cone.get("data", [])
    rows = filter_obs(rows, mission)
    seq = sector if mission.lower() == "tess" else quarter
    chosen = choose_obs(rows, seq)
    obsid = int(chosen["obsid"])

    # Get products for this obsid
    prod = caom_products(obsid)  # Mast.Caom.Products
    plist = prod.get("data", [])
    if not plist:
        raise FileNotFoundError(f"No products for obsid={obsid}.")

    product = pick_one_product(plist, prefer=prefer)
    dataURI = product["dataURI"]  # e.g., 'mast:TESS/product/…fits'
    fname = os.path.basename(product["productFilename"]) or "download.fits"

    # Organize path: e.g., data/TESS/<obs_id>/<filename>
    local_path = outdir / (product.get("obs_collection") or mission.upper()) / (product.get("obs_id") or str(obsid))
    local_path.mkdir(parents=True, exist_ok=True)
    local_path = local_path / fname

    # Download exactly one file
    download_request(dataURI, filename=str(local_path), download_type="file")
    print(f"Saved: {local_path}")
    return local_path

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Download ONE FITS (latest sector/quarter) for a TESS/Kepler target from MAST.")
    ap.add_argument("--mission", required=True, choices=["tess", "kepler"], help="Mission")
    ap.add_argument("--id", required=True, help="Target ID (TIC/KIC or numeric)")
    ap.add_argument("--sector", type=int, default=None, help="TESS sector to force (else latest)")
    ap.add_argument("--quarter", type=int, default=None, help="Kepler quarter to force (else latest)")
    ap.add_argument("--prefer", choices=["lc", "tpf", "any"], default="lc", help="Prefer lightcurve or TPF")
    ap.add_argument("--out", default="./data", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        download_one(
            mission=args.mission,
            target_id=args.id,
            sector=args.sector,
            quarter=args.quarter,
            outdir=outdir,
            prefer=("any" if args.prefer == "any" else args.prefer),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
