#!/usr/bin/env python3
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=".*tpfmodel submodule is not available.*",
    category=UserWarning,
)

import lightkurve as lk  
from astroquery.mast import Observations


class DownloadError(RuntimeError):
    pass


@dataclass
class LatestFITSFetcher:
    """
    Fetch exactly ONE FITS file (latest by sector/quarter or time) for a given target.

    - mission: 'tess' or 'kepler' (case-insensitive)
    - target_id: TIC or KIC identifier; numeric strings allowed
    - outdir: where to download; created if missing

    TESS options:
      - sector: optional int; if provided we filter for that sector (or sequence_number fallback)
      - tesscut: if True, fetch FFI cutout via TESSCut instead of SPOC/QLP lightcurve
      - cutout_size: TESSCut square size in pixels
      - prefer: 'spoc', 'qlp', or 'any' (affects author priority for TESS lightcurves)

    Kepler options:
      - quarter: optional int; if provided we filter for that quarter (or campaign fallback)

    Call .download() -> Path to the saved FITS.
    """

    mission: str
    target_id: str
    outdir: Path | str = "./tmp"

    # TESS-specific
    sector: Optional[int] = None
    tesscut: bool = False
    cutout_size: int = 31
    prefer: str = "spoc"  # 'spoc' | 'qlp' | 'any'

    # Kepler-specific
    quarter: Optional[int] = None

    # Misc
    verbose: bool = False

    _author_priority: Tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self):
        self.mission = self.mission.strip().lower()
        if self.mission not in ("tess", "kepler"):
            raise ValueError("mission must be 'tess' or 'kepler'")

        self.outdir = Path(self.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

        pref = (self.prefer or "spoc").strip().lower()
        if pref not in ("spoc", "qlp", "any"):
            raise ValueError("prefer must be 'spoc', 'qlp', or 'any'")
        self.prefer = pref
        self._author_priority = {
            "spoc": ("SPOC", "QLP"),
            "qlp": ("QLP", "SPOC"),
            "any": (),
        }[self.prefer]

    # -------- Public API -------- #

    def download(self) -> Path:
        """Download exactly one FITS for the configured target/options."""
        if self.mission == "tess":
            if self.tesscut:
                return self._download_one_tesscut()
            return self._download_one_tess_lightcurve()
        else:  # kepler
            return self._download_one_kepler_lightcurve()

    # -------- Internals -------- #

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _target_name(mission: str, target_id: str) -> str:
        m = mission.strip().lower()
        tid = target_id.strip()
        if m == "tess":
            return tid if tid.lower().startswith("tic") else f"TIC {tid}"
        if m == "kepler":
            return tid if tid.lower().startswith("kic") else f"KIC {tid}"
        raise ValueError("mission must be 'tess' or 'kepler'")

    @staticmethod
    def _available_columns(sr: "lk.search.SearchResult") -> set:
        try:
            return set(sr.table.colnames)
        except Exception:
            return set()

    def _pick_one_row_dynamic(
        self,
        sr: "lk.search.SearchResult",
        mission: str,
        value: Optional[int],
        author_priority: Sequence[str],
    ) -> "lk.search.SearchResult":
        """Robustly pick exactly one row.

        Strategy:
        - Grouping columns:
            TESS: ['sector', 'sequence_number']
            Kepler: ['quarter', 'campaign']
        - If `value` provided, filter on first available grouping column.
        - Else, choose max over first available grouping column.
        - Apply author priority if present.
        - Sort by observation_time_start (or 'time') and take the latest.
        """
        if len(sr) == 0:
            raise FileNotFoundError("No search results found.")

        mission = mission.lower().strip()
        cols = self._available_columns(sr)
        if mission == "tess":
            group_cols = ["sector", "sequence_number"]
            value_label = "sector"
        elif mission == "kepler":
            group_cols = ["quarter", "campaign"]
            value_label = "quarter"
        else:
            group_cols, value_label = [], "group"

        by_col = next((c for c in group_cols if c in cols), None)

        sr2 = sr
        chosen_value = None

        if value is not None:
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
            try:
                chosen_value = int(max(sr.table[by_col]))
                sr2 = sr[sr.table[by_col] == chosen_value]
            except Exception:
                sr2 = sr  # fall back to time sort

        # Author priority
        if "author" in self._available_columns(sr2) and author_priority:
            for auth in author_priority:
                cand = sr2[sr2.table["author"] == auth]
                if len(cand) > 0:
                    sr2 = cand
                    break

        # Sort by time and take the latest
        sort_col = None
        if "observation_time_start" in self._available_columns(sr2):
            sort_col = "observation_time_start"
        elif "time" in self._available_columns(sr2):
            sort_col = "time"

        if sort_col:
            sr2 = sr2.sort(sort_col)

        if len(sr2) == 0:
            raise FileNotFoundError("No results left after filtering.")

        chosen = sr2[-1]  # single-row SearchResult

        # Debug log
        try:
            bits = []
            if by_col is not None and chosen_value is not None:
                bits.append(f"{by_col}={chosen_value}")
            if "author" in chosen.table.colnames:
                bits.append(f"author={chosen.table['author'][0]}")
            if "productFilename" in chosen.table.colnames:
                bits.append(f"file={chosen.table['productFilename'][0]}")
            self._log("Chosen product: " + "; ".join(bits))
        except Exception:
            pass

        return chosen

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

    # ---- TESS ---- #

    def _download_one_tess_lightcurve(self) -> Path:
        tname = self._target_name("tess", self.target_id)
        sr = lk.search_lightcurve(tname, mission="TESS")
        row = self._pick_one_row_dynamic(
            sr, mission="tess", value=self.sector, author_priority=self._author_priority
        )

        lc = row.download(download_dir=str(self.outdir))

        expected = None
        try:
            expected = lc.meta.get("FILENAME")
        except Exception:
            pass

        saved = self._find_actual_download(self.outdir, expected_basename=expected)
        if not saved:
            raise DownloadError("Download completed but FITS not found on disk.")
        return saved

    def _download_one_tesscut(self) -> Path:
        tname = self._target_name("tess", self.target_id)
        sr = lk.search_tesscut(tname)
        row = self._pick_one_row_dynamic(
            sr, mission="tess", value=self.sector, author_priority=()
        )

        tpf = row.download(cutout_size=int(self.cutout_size), download_dir=str(self.outdir))

        expected = None
        try:
            expected = tpf.hdu[0].header.get("FILENAME")
        except Exception:
            pass

        saved = self._find_actual_download(self.outdir, expected_basename=expected)
        if not saved:
            raise DownloadError("TESSCut completed but FITS not found on disk.")
        return saved

    # ---- Kepler ---- #

    def _download_one_kepler_lightcurve(self) -> Path:
        tname = self._target_name("kepler", self.target_id)
        sr = lk.search_lightcurve(tname, mission="Kepler")
        row = self._pick_one_row_dynamic(
            sr, mission="kepler", value=self.quarter, author_priority=()
        )

        lc = row.download(download_dir=str(self.outdir))

        expected = None
        try:
            expected = lc.meta.get("FILENAME")
        except Exception:
            pass

        saved = self._find_actual_download(self.outdir, expected_basename=expected)
        if not saved:
            raise DownloadError("Download completed but FITS not found on disk.")
        return saved




    # TESSCut (specific sector, 31px cutout)
    # fetcher = LatestFITSFetcher(mission="tess", target_id="25155310", sector=42, tesscut=True, cutout_size=31)
    # print(fetcher.download())

    # Kepler (latest quarter)
    # fetcher = LatestFITSFetcher(mission="kepler", target_id="11446443", outdir="./data")
    # print(fetcher.download())
