# src/stellar_data_fetcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math, re

from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


class StarNotFound(Exception):
    pass


@dataclass
class StarMetaFetcher:
    """
    Fast, no-download stellar metadata fetcher.

    - mission: 'tess' or 'kepler'
    - target_id: TIC / TOI / KIC / KOI (flexible parsing)

    Public:
      fetch_stellar() -> {"source": <str>, "stellar": {st_*}, "raw": <dict>}
    """
    mission: str
    target_id: str

    def fetch_stellar(self) -> Dict[str, Any]:
        m = (self.mission or "").strip().lower()
        ids = self._parse_ids()

        if m == "tess":
            # Resolve TOI -> TIC if needed via NEA TOI table
            tic = ids.get("tic")
            if tic is None and ids.get("toi") is not None:
                toi_like = ids["toi"]  # e.g., "700.01" or "K00700.01"
                q = NasaExoplanetArchive.query_advanced(
                    table="toi",
                    select="tid",
                    where=f"toi={toi_like}" if toi_like.replace('.', '', 1).isdigit()
                          else f"koi LIKE '{toi_like}%'"
                )
                if len(q) == 0:
                    raise StarNotFound(f"TOI {toi_like} not found in NEA.")
                tic = int(q["tid"][0])

            if tic is None:
                raise StarNotFound("Could not resolve a TIC id.")

            # Query TIC via MAST Catalogs (no file download)
            tab = Catalogs.query_object(f"TIC {tic}", catalog="TIC")
            if len(tab) == 0:
                raise StarNotFound(f"TIC {tic} not found in TIC.")
            row = {c: self._to_py(tab[0][c]) for c in tab.colnames}
            return {
                "source": "MAST TIC",
                "stellar": self._map_tic_to_st(row),
                "raw": row,
            }

        if m == "kepler":
            kepid = ids.get("kepid")
            if kepid is None and ids.get("koi") is not None:
                koi = ids["koi"]  # e.g., "K00700.01" or "K00700"
                q = NasaExoplanetArchive.query_advanced(
                    table="cumulative", select="kepid", where=f"kepoi_name LIKE '{koi}%'"
                )
                if len(q) == 0:
                    raise StarNotFound(f"KOI {koi} not found in NEA cumulative.")
                kepid = int(q["kepid"][0])

            if kepid is None:
                raise StarNotFound("Could not resolve a KIC id.")

            # DR25 stellar parameters (no file download)
            dr = NasaExoplanetArchive.query_advanced(
                table="q1_q17_dr25_stellar", where=f"kepid={int(kepid)}"
            )
            if len(dr) == 0:
                raise StarNotFound(f"DR25 stellar not found for KIC {kepid}.")
            row = {c: self._to_py(dr[0][c]) for c in dr.colnames}
            return {
                "source": "NEA Kepler DR25 stellar",
                "stellar": self._map_dr25_to_st(row),
                "raw": row,
            }

        raise ValueError("mission must be 'tess' or 'kepler'")

    # ----------------- helpers -----------------

    def _parse_ids(self) -> Dict[str, Optional[int | str]]:
        """Accept TIC/TOI/KIC/KOI styles and normalize."""
        m = (self.mission or "").strip().lower()
        t = (self.target_id or "").strip()

        if m == "tess":
            # TIC nnnn or "TIC nnnn"
            mtic = re.search(r"(?i)(?:tic[:\s-]*)?(\d+)", t)
            if mtic:
                return {"tic": int(mtic.group(1)), "toi": None}
            # TOI-1234.01 / TOI 1234.01
            mtoi = re.search(r"(?i)toi[\s-]*([0-9]+(?:\.[0-9]+)?)", t)
            if mtoi:
                return {"tic": None, "toi": mtoi.group(1)}
            return {"tic": None, "toi": None}

        if m == "kepler":
            # KOI-1234.01 or K00001.01 -> KOI code
            mkoi = re.search(r"(?i)k(?:oi[-\s]*)?(\d+)(?:[.-](\d+))?", t)
            if mkoi:
                base = int(mkoi.group(1))
                comp = mkoi.group(2)
                koi = f"K{base:05d}" + (f".{int(comp):02d}" if comp else "")
                return {"koi": koi, "kepid": None}
            # KIC nnnn
            mkic = re.search(r"(?i)(?:kic[:\s-]*)?(\d+)", t)
            if mkic:
                return {"koi": None, "kepid": int(mkic.group(1))}
            return {"koi": None, "kepid": None}

        return {}

    @staticmethod
    def _to_py(v):
        try:
            return v.item()
        except Exception:
            return v

    @staticmethod
    def _pick(row: Dict[str, Any], *names) -> Optional[float]:
        for n in names:
            if not n:
                continue
            v = row.get(n)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            try:
                return float(v)
            except Exception:
                continue
        return None

    # ---- mappers ----

    def _map_tic_to_st(self, row: Dict[str, Any]) -> Dict[str, Optional[float]]:
        # TIC columns commonly: Teff, e_Teff, logg, e_logg, rad, e_rad, mass, e_mass,
        # Tmag, e_Tmag, d, e_d (distance); not all rows have all fields.
        pk = self._pick
        eT = pk(row, "e_Teff")
        eR = pk(row, "e_rad")
        eM = pk(row, "e_mass")
        eG = pk(row, "e_logg")
        eD = pk(row, "e_d")
        eTm = pk(row, "e_Tmag")

        return {
            "st_teff":     pk(row, "Teff"),
            "st_tefferr1": eT,
            "st_tefferr2": (-eT if eT is not None else None),
            "st_rad":      pk(row, "rad"),
            "st_raderr1":  eR,
            "st_raderr2":  (-eR if eR is not None else None),
            "st_mass":     pk(row, "mass"),
            "st_masserr1": eM,
            "st_masserr2": (-eM if eM is not None else None),
            "st_logg":     pk(row, "logg"),
            "st_loggerr1": eG,
            "st_loggerr2": (-eG if eG is not None else None),
            "st_dist":     pk(row, "d"),
            "st_disterr1": eD,
            "st_disterr2": (-eD if eD is not None else None),
            "st_tmag":     pk(row, "Tmag"),
            "st_tmagerr1": eTm,
            "st_tmagerr2": (-eTm if eTm is not None else None),
        }

    def _map_dr25_to_st(self, row: Dict[str, Any]) -> Dict[str, Optional[float]]:
        # DR25 names vary slightly; try multiple candidates.
        pk = self._pick

        te   = pk(row, "teff", "kic_teff")
        te1  = pk(row, "teff_err1", "kic_teff_err1", "teff_err")
        te2  = pk(row, "teff_err2", "kic_teff_err2")

        lg   = pk(row, "logg", "kic_logg")
        lg1  = pk(row, "logg_err1", "kic_logg_err1", "logg_err")
        lg2  = pk(row, "logg_err2", "kic_logg_err2")

        rad  = pk(row, "radius", "kic_radius", "koi_srad")
        rad1 = pk(row, "radius_err1", "kic_radius_err1", "radius_err")
        rad2 = pk(row, "radius_err2", "kic_radius_err2")

        mas  = pk(row, "mass", "kic_mass")
        mas1 = pk(row, "mass_err1", "kic_mass_err1", "mass_err")
        mas2 = pk(row, "mass_err2", "kic_mass_err2")

        dist = pk(row, "dist", "kic_dist", "d")
        d1   = pk(row, "dist_err1", "kic_dist_err1", "e_d")
        d2   = pk(row, "dist_err2", "kic_dist_err2")

        return {
            "st_teff":      te,
            "st_tefferr1":  te1,
            "st_tefferr2":  te2 if te2 is not None else (-te1 if te1 is not None else None),
            "st_rad":       rad,
            "st_raderr1":   rad1,
            "st_raderr2":   rad2 if rad2 is not None else (-rad1 if rad1 is not None else None),
            "st_mass":      mas,
            "st_masserr1":  mas1,
            "st_masserr2":  mas2 if mas2 is not None else (-mas1 if mas1 is not None else None),
            "st_logg":      lg,
            "st_loggerr1":  lg1,
            "st_loggerr2":  lg2 if lg2 is not None else (-lg1 if lg1 is not None else None),
            "st_dist":      dist,
            "st_disterr1":  d1,
            "st_disterr2":  d2 if d2 is not None else (-d1 if d1 is not None else None),
            "st_tmag":      None,   # Kepler stars don't have TESS mag
            "st_tmagerr1":  None,
            "st_tmagerr2":  None,
        }
