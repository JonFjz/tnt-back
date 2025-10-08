# latest_fits_fetcher_optimized.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
import json, time

import lightkurve as lk
from astroquery.mast import Observations


class DownloadError(RuntimeError):
    pass


@dataclass
class LatestFITSFetcher:
    """
    Download (or select) ONE best lightcurve product for a target.

    Modes:
      - headers_only=True  -> no download; return chosen product meta
      - tesscut=True       -> explicit FFI cutout (heavier)
      - prefer_flux: 'pdcsap' | 'sap' | 'any' (advisory; applied after download)

    Cached manifest stored in `manifest.json` inside outdir to reduce repeat work.
    """
    mission: str
    target_id: str
    outdir: Path | str = "./tmp"

    # TESS-specific
    sector: Optional[int] = None
    tesscut: bool = False
    cutout_size: int = 31
    prefer: str = "spoc"        # 'spoc' | 'qlp' | 'any'
    prefer_flux: str = "pdcsap" # 'pdcsap' | 'sap' | 'any'

    # Kepler-specific
    quarter: Optional[int] = None

    # Behavior
    headers_only: bool = False
    verbose: bool = False

    _author_priority: Tuple[str, ...] = field(init=False, repr=False)
    _manifest_path: Path = field(init=False, repr=False)

    def __post_init__(self):
        self.mission = self.mission.strip().lower()
        if self.mission not in ("tess", "kepler"):
            raise ValueError("mission must be 'tess' or 'kepler'")

        self.outdir = Path(self.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.outdir / "manifest.json"

        pref = (self.prefer or "spoc").strip().lower()
        if pref not in ("spoc", "qlp", "any"):
            raise ValueError("prefer must be 'spoc', 'qlp', or 'any'")
        self.prefer = pref
        self._author_priority = {
            "spoc": ("SPOC", "QLP"),
            "qlp": ("QLP", "SPOC"),
            "any": (),
        }[self.prefer]

        self.prefer_flux = (self.prefer_flux or "pdcsap").lower()
        if self.prefer_flux not in ("pdcsap", "sap", "any"):
            raise ValueError("prefer_flux must be 'pdcsap', 'sap', or 'any'")

    # ----------------------------- public ------------------------------------

    def select_or_download(self) -> Dict[str, Any]:
        """
        Returns a dict with selection details, plus local 'path' if downloaded.
        {
          "mission", "target", "mode", "author", "group_value",
          "productFilename", "path"?, "notes"?, "meta": {...}
        }
        """
        if self.mission == "tess":
            if self.tesscut:
                return self._tess_tesscut()
            return self._tess_lightcurve()
        else:
            return self._kepler_lightcurve()

    # ----------------------------- internals ---------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _target_name(self, mission: str, target_id: str) -> str:
        m = mission.strip().lower()
        tid = target_id.strip()
        if m == "tess":
            return tid if tid.lower().startswith("tic") else f"TIC {tid}"
        if m == "kepler":
            return tid if tid.lower().startswith("kic") else f"KIC {tid}"
        raise ValueError("mission must be 'tess' or 'kepler'")

    # -------- manifest cache --------

    def _manifest_load(self) -> dict:
        try:
            if self._manifest_path.exists():
                return json.loads(self._manifest_path.read_text())
        except Exception:
            pass
        return {}

    def _manifest_save(self, data: dict):
        try:
            self._manifest_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _manifest_key(self, **kwargs) -> str:
        return json.dumps({
            "mission": self.mission,
            "target": self.target_id,
            "sector": self.sector,
            "quarter": self.quarter,
            "prefer": self.prefer,
            "flux": self.prefer_flux,
            "tesscut": self.tesscut,
            "headers_only": self.headers_only,
        }, sort_keys=True)

    # -------- selection core --------

    def _pick_one_row_dynamic(self, sr: "lk.search.SearchResult",
                              mission: str,
                              value: Optional[int],
                              author_priority: Sequence[str]):
        if len(sr) == 0:
            raise FileNotFoundError("No search results found.")
        cols = set(getattr(sr.table, "colnames", []))
        # grouping columns
        if mission == "tess":
            groups = ["sector", "sequence_number"]
            label = "sector"
        else:
            groups = ["quarter", "campaign"]
            label = "quarter"

        by_col = next((c for c in groups if c in cols), None)
        chosen = sr
        chosen_value = None

        if value is not None:
            for c in groups:
                if c in cols:
                    chosen = sr[sr.table[c] == int(value)]
                    by_col = c
                    chosen_value = int(value)
                    break
            if len(chosen) == 0:
                raise FileNotFoundError(f"No results for requested {label}={value}.")

        elif by_col is not None:
            try:
                chosen_value = int(max(sr.table[by_col]))
                chosen = sr[sr.table[by_col] == chosen_value]
            except Exception:
                pass

        # author priority
        if "author" in cols and author_priority:
            for a in author_priority:
                cand = chosen[chosen.table["author"] == a]
                if len(cand) > 0:
                    chosen = cand
                    break

        # latest by observation_time_start if present
        sort_col = "observation_time_start" if "observation_time_start" in cols else None
        if sort_col:
            chosen = chosen.sort(sort_col)

        row = chosen[-1]
        return row, by_col, chosen_value

    def _tess_lightcurve(self) -> Dict[str, Any]:
        tname = self._target_name("tess", self.target_id)
        sr = lk.search_lightcurve(tname, mission="TESS")
        row, by_col, chosen_value = self._pick_one_row_dynamic(
            sr, mission="tess", value=self.sector, author_priority=self._author_priority
        )

        # manifest cache
        key = self._manifest_key()
        man = self._manifest_load()
        if key in man and self.headers_only:
            return man[key]

        meta = {
            "author": str(row.table["author"][0]) if "author" in row.table.colnames else None,
            "group_col": by_col,
            "group_val": chosen_value,
            "productFilename": str(row.table["productFilename"][0]) if "productFilename" in row.table.colnames else None,
        }

        if self.headers_only:
            result = {
                "mission": "tess",
                "target": tname,
                "mode": "headers_only",
                **meta
            }
            man[key] = result
            self._manifest_save(man)
            return result

        lc = row.download(download_dir=str(self.outdir))
        # Prefer PDCSAP if available in the file (selection used later in processing)
        result = {
            "mission": "tess",
            "target": tname,
            "mode": "download",
            **meta,
            "path": str(self._find_actual_download(self.outdir, getattr(lc, "filename", None))),
            "notes": f"prefer_flux={self.prefer_flux}"
        }
        man[key] = result
        self._manifest_save(man)
        return result

    def _tess_tesscut(self) -> Dict[str, Any]:
        tname = self._target_name("tess", self.target_id)
        sr = lk.search_tesscut(tname)
        row, by_col, chosen_value = self._pick_one_row_dynamic(
            sr, mission="tess", value=self.sector, author_priority=()
        )

        key = self._manifest_key()
        man = self._manifest_load()
        if key in man and self.headers_only:
            return man[key]

        if self.headers_only:
            result = {
                "mission": "tess",
                "target": tname,
                "mode": "headers_only_tesscut",
                "group_col": by_col,
                "group_val": chosen_value,
            }
            man[key] = result
            self._manifest_save(man)
            return result

        tpf = row.download(cutout_size=int(self.cutout_size), download_dir=str(self.outdir))
        result = {
            "mission": "tess",
            "target": tname,
            "mode": "download_tesscut",
            "group_col": by_col,
            "group_val": chosen_value,
            "path": str(self._find_actual_download(self.outdir, getattr(tpf, "filename", None))),
        }
        man[key] = result
        self._manifest_save(man)
        return result

    def _kepler_lightcurve(self) -> Dict[str, Any]:
        tname = self._target_name("kepler", self.target_id)
        sr = lk.search_lightcurve(tname, mission="Kepler")
        row, by_col, chosen_value = self._pick_one_row_dynamic(
            sr, mission="kepler", value=self.quarter, author_priority=()
        )

        key = self._manifest_key()
        man = self._manifest_load()
        if key in man and self.headers_only:
            return man[key]

        meta = {
            "author": str(row.table["author"][0]) if "author" in row.table.colnames else None,
            "group_col": by_col,
            "group_val": chosen_value,
            "productFilename": str(row.table["productFilename"][0]) if "productFilename" in row.table.colnames else None,
        }

        if self.headers_only:
            result = {
                "mission": "kepler",
                "target": tname,
                "mode": "headers_only",
                **meta
            }
            man[key] = result
            self._manifest_save(man)
            return result

        lc = row.download(download_dir=str(self.outdir))
        result = {
            "mission": "kepler",
            "target": tname,
            "mode": "download",
            **meta,
            "path": str(self._find_actual_download(self.outdir, getattr(lc, "filename", None))),
            "notes": f"prefer_flux={self.prefer_flux}"
        }
        man[key] = result
        self._manifest_save(man)
        return result

    @staticmethod
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
