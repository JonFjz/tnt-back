# star_meta_fetcher_optimized.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, Tuple
import re, math, time, os
from collections import OrderedDict

from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


# --------------------------- small, fast TTL cache ---------------------------

class TTLCache:
    """Simple in-process TTL+LRU cache."""
    def __init__(self, maxsize: int = 512, ttl: float = 3600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        now = time.time()
        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < now:
            # expired -> drop
            self._data.pop(key, None)
            return None
        # LRU touch
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        now = time.time()
        self._data[key] = (now + self.ttl, value)
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)


# Optional Redis backend (only if you want it; otherwise ignore)
class RedisCache:
    """Tiny wrapper; expects a redis-py client with get/setex."""
    def __init__(self, client, ttl: int = 3600):
        self.client = client
        self.ttl = ttl

    def get(self, key: str) -> Any | None:
        import json
        raw = self.client.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any):
        import json
        self.client.setex(key, self.ttl, json.dumps(value))


# ----------------------------- fetcher proper -------------------------------

class StarNotFound(Exception):
    pass


@dataclass
class StarMetaFetcher:
    """
    Fast, cached stellar metadata fetcher for TESS/Kepler.

    Public:
      fetch_stellar() -> {
        "source": <str>,
        "stellar": {st_* fields...},
        "raw": <dict of raw row>,
        "timing_ms": <float>
      }

    Notes:
      - Only the columns needed by your models are requested where possible.
      - Caches both ID resolution (TOI->TIC, KOI->KIC) and final stellar rows.
    """
    mission: str
    target_id: str
    cache: Any | None = None     # TTLCache or RedisCache or None
    cache_ttl_s: int = 3600

    # Columns we actually need downstream (TIC / DR25 names vary; we trim after fetch)
    TIC_SELECT = [
        "Teff","e_Teff","logg","e_logg","rad","e_rad","mass","e_mass",
        "Tmag","e_Tmag","d","e_d"
    ]
    # DR25 columns (attempt these first; we have fallbacks in the mapper)
    DR25_SELECT = [
        "kepid","teff","teff_err1","teff_err2",
        "logg","logg_err1","logg_err2",
        "radius","radius_err1","radius_err2",
        "mass","mass_err1","mass_err2",
        "dist","dist_err1","dist_err2"
    ]

    def _cache(self) -> Any:
        if self.cache is None:
            # default lightweight cache
            self.cache = TTLCache(maxsize=1024, ttl=self.cache_ttl_s)
        return self.cache

    # ----------------------------- public API --------------------------------

    def fetch_stellar(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        m = (self.mission or "").strip().lower()
        if m not in ("tess", "kepler"):
            raise ValueError("mission must be 'tess' or 'kepler'")

        # normalize + cached ID resolution
        ids_key = f"ids:{m}:{self.target_id.strip().lower()}"
        ids = self._cache().get(ids_key)
        if ids is None:
            ids = self._parse_and_resolve_ids(m, self.target_id)
            self._cache().set(ids_key, ids)

        if m == "tess":
            tic = ids.get("tic")
            if tic is None:
                raise StarNotFound("Could not resolve a TIC id.")
            row, source = self._query_tic_row(tic)
            stellar = self._map_tic_to_st(row)

        else:  # kepler
            kepid = ids.get("kepid")
            if kepid is None:
                raise StarNotFound("Could not resolve a KIC id.")
            row, source = self._query_dr25_row(kepid)
            stellar = self._map_dr25_to_st(row)

        t1 = time.perf_counter()
        return {
            "source": source,
            "stellar": stellar,
            "raw": row,
            "timing_ms": round((t1 - t0) * 1000.0, 2),
        }

    # --------------------------- ID normalization ----------------------------

    def _parse_and_resolve_ids(self, mission: str, target: str) -> Dict[str, Any]:
        t = (target or "").strip()

        if mission == "tess":
            # TIC nnnn or TOI-x.y
            mtic = re.search(r"(?i)(?:tic[:\s-]*)?(\d+)", t)
            if mtic:
                return {"tic": int(mtic.group(1))}
            mtoi = re.search(r"(?i)toi[\s-]*([0-9]+(?:\.[0-9]+)?)", t)
            if mtoi:
                toi_val = mtoi.group(1)
                # NEA: resolve TOI -> TIC id (tid)
                q = NasaExoplanetArchive.query_advanced(
                    table="toi", select="tid", where=f"toi={toi_val}"
                )
                if len(q) == 0 or q["tid"][0] in (None, ""):
                    raise StarNotFound(f"TOI {toi_val} not found.")
                return {"tic": int(q["tid"][0])}
            raise StarNotFound("Provide a TIC ID or TOI label.")

        else:
            # Kepler: KIC nnnn or KOI (K00001.01 / KOI-1.01 etc)
            mkic = re.search(r"(?i)(?:kic[:\s-]*)?(\d+)", t)
            if mkic:
                return {"kepid": int(mkic.group(1))}
            mkoi = re.search(r"(?i)k(?:oi[-\s]*)?(\d+)(?:[.-](\d+))?", t)
            if mkoi:
                base = int(mkoi.group(1))
                comp = mkoi.group(2)
                koi = f"K{base:05d}" + (f".{int(comp):02d}" if comp else "")
                q = NasaExoplanetArchive.query_advanced(
                    table="cumulative", select="kepid", where=f"kepoi_name LIKE '{koi}%'"
                )
                if len(q) == 0 or q["kepid"][0] in (None, ""):
                    raise StarNotFound(f"KOI {koi} not found.")
                return {"kepid": int(q["kepid"][0])}
            raise StarNotFound("Provide a KIC ID or KOI label.")

    # --------------------------- remote queries ------------------------------

    def _query_tic_row(self, tic: int) -> tuple[Dict[str, Any], str]:
        cache_key = f"tic:{tic}"
        cached = self._cache().get(cache_key)
        if cached:
            return cached, "MAST TIC (cache)"

        # Try server-side select (faster). Fallback trims client-side.
        row = None
        try:
            tab = Catalogs.query_criteria(
                catalog="TIC", ID=int(tic), select=",".join(self.TIC_SELECT)
            )
            if len(tab) > 0:
                row = {c: self._to_py(tab[0][c]) for c in tab.colnames}
        except Exception:
            pass

        if row is None:
            tab = Catalogs.query_object(f"TIC {tic}", catalog="TIC")
            if len(tab) == 0:
                raise StarNotFound(f"TIC {tic} not found in TIC.")
            raw = {c: self._to_py(tab[0][c]) for c in tab.colnames}
            # trim only needed keys if present
            row = {k: raw.get(k) for k in self.TIC_SELECT if k in raw}
            # keep originals for mappers too
            row |= raw

        self._cache().set(cache_key, row)
        return row, "MAST TIC"

    def _query_dr25_row(self, kepid: int) -> tuple[Dict[str, Any], str]:
        cache_key = f"dr25:{kepid}"
        cached = self._cache().get(cache_key)
        if cached:
            return cached, "NEA DR25 (cache)"

        row = None
        try:
            dr = NasaExoplanetArchive.query_advanced(
                table="q1_q17_dr25_stellar",
                select=",".join(self.DR25_SELECT),
                where=f"kepid={int(kepid)}"
            )
            if len(dr) > 0:
                row = {c: self._to_py(dr[0][c]) for c in dr.colnames}
        except Exception:
            pass

        if row is None:
            dr = NasaExoplanetArchive.query_advanced(
                table="q1_q17_dr25_stellar", where=f"kepid={int(kepid)}"
            )
            if len(dr) == 0:
                raise StarNotFound(f"DR25 stellar not found for KIC {kepid}.")
            raw = {c: self._to_py(dr[0][c]) for c in dr.colnames}
            # Keep only useful ones if present
            trimmed = {k: raw.get(k) for k in self.DR25_SELECT if k in raw}
            row = trimmed | raw

        self._cache().set(cache_key, row)
        return row, "NEA Kepler DR25 stellar"

    # ------------------------------ mappers ----------------------------------

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

    def _map_tic_to_st(self, row: Dict[str, Any]) -> Dict[str, Optional[float]]:
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
            "st_tmag":      None,
            "st_tmagerr1":  None,
            "st_tmagerr2":  None,
        }
