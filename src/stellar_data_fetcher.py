# pip install astroquery astropy numpy
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

from astroquery.mast import Catalogs, Observations
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


class StarNotFound(Exception):
    pass


@dataclass
class StarMetaFetcher:
    """
    Fetch stellar metadata and enumerate MAST products for a target.

    - TESS  -> TIC (via MAST Catalogs)
    - Kepler-> DR25 stellar table (via NASA Exoplanet Archive; authoritative vs. old KIC)

    Usage:
        meta = StarMetaFetcher("tess", "25155310")
        stellar = meta.fetch_stellar()         # normalized st_* + raw source row
        products = meta.list_mast_products()   # all MAST products for this target
    """
    mission: str   # 'tess' or 'kepler'
    target_id: Union[str, int]
    verbose: bool = False

    def __post_init__(self):
        m = (self.mission or "").strip().lower()
        if m not in {"tess", "kepler"}:
            raise ValueError("mission must be 'TESS' or 'Kepler'")
        self.mission = m

    # ------------------------- Public API -------------------------

    def fetch_stellar(self) -> Dict[str, Any]:
        """Return normalized stellar dict (st_*) and raw source row."""
        return self._fetch_tic() if self.mission == "tess" else self._fetch_kepler_dr25()

    def list_mast_products(self) -> List[Dict[str, Any]]:
        """
        List all MAST products for this target (LCs, TPFs, DV reports/TS, etc.).
        Keeps key columns, JSON-safe.
        """
        name = self._target_name(self.mission, self.target_id)
        obs = Observations.query_object(name, radius="0.001 deg")  # tight cone
        if len(obs) == 0:
            return []

        prods = Observations.get_product_list(obs)
        keep = {
            "obsid", "target_name", "obs_collection", "instrument_name",
            "productType", "productFilename", "description", "dataURI"
        }

        out: List[Dict[str, Any]] = []
        for row in prods:
            d: Dict[str, Any] = {}
            for col in row.colnames:
                if col not in keep:
                    continue
                val = row[col]
                # Handle masked and numpy scalars
                if hasattr(val, "mask"):
                    if np.all(getattr(val, "mask")):
                        val = None
                    else:
                        val = val.filled(np.nan)
                try:
                    val = val.item()
                except Exception:
                    pass
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val = None
                d[col] = val
            out.append(d)
        return out

    # ----------------------- Internal: TESS -----------------------

    def _fetch_tic(self) -> Dict[str, Any]:
        tic = self._parse_int_id(self.target_id, prefix="TIC")
        tab = Catalogs.query_object(f"TIC {tic}", catalog="TIC")
        if len(tab) == 0:
            raise StarNotFound(f"TIC {tic} not found")

        r = tab[0]
        raw = self._row_to_dict(r)

        # TIC provides symmetric uncertainties e_* for many fields.
        # We duplicate them into err1/err2 to match your schema.
        st = {
            "catalog": "TIC",
            "tic_id": tic,
            "ra": self._get(r, "ra"),
            "dec": self._get(r, "dec"),

            "st_teff":      self._get(r, "Teff"),
            "st_tefferr1":  self._get(r, "e_Teff"),
            "st_tefferr2":  self._get(r, "e_Teff"),

            "st_logg":      self._get(r, "logg"),
            "st_loggerr1":  self._get(r, "e_logg"),
            "st_loggerr2":  self._get(r, "e_logg"),

            "st_rad":       self._get(r, "rad"),
            "st_raderr1":   self._get(r, "e_rad"),
            "st_raderr2":   self._get(r, "e_rad"),

            "st_mass":      self._get(r, "mass") if "mass" in r.colnames else None,
            "st_masserr1":  self._get(r, "e_mass") if "e_mass" in r.colnames else None,
            "st_masserr2":  self._get(r, "e_mass") if "e_mass" in r.colnames else None,

            "st_dist":      self._get(r, "d"),
            "st_disterr1":  self._get(r, "e_d"),
            "st_disterr2":  self._get(r, "e_d"),

            "st_tmag":      self._get(r, "Tmag"),
            "st_tmagerr1":  self._get(r, "e_Tmag") if "e_Tmag" in r.colnames else None,
            "st_tmagerr2":  self._get(r, "e_Tmag") if "e_Tmag" in r.colnames else None,

            "st_pmra":      self._get(r, "pmRA"),
            "st_pmraerr":   self._get(r, "e_pmRA") if "e_pmRA" in r.colnames else None,
            "st_pmdec":     self._get(r, "pmDEC"),
            "st_pmdecerr":  self._get(r, "e_pmDEC") if "e_pmDEC" in r.colnames else None,
        }

        return {"stellar": st, "source_row": raw}

    # ---------------------- Internal: Kepler ----------------------

    def _fetch_kepler_dr25(self) -> Dict[str, Any]:
        kic = self._parse_int_id(self.target_id, prefix="KIC")
        tab = NasaExoplanetArchive.query_criteria(
            table="q1_q17_dr25_stellar",
            select=("kepid,ra,dec,teff,teff_err1,teff_err2,logg,logg_err1,logg_err2,"
                    "radius,radius_err1,radius_err2,dist,dist_err1,dist_err2,"
                    "kepmag,pmra,pmdec"),
            where=f"kepid={kic}",
        )
        if len(tab) == 0:
            raise StarNotFound(f"KIC {kic} not found in Kepler DR25 stellar table")

        r = tab[0]
        raw = self._row_to_dict(r)

        st = {
            "catalog": "Kepler DR25",
            "kepid": int(r["kepid"]),
            "ra": self._as_float(r["ra"]),
            "dec": self._as_float(r["dec"]),

            "st_teff":      self._get(r, "teff"),
            "st_tefferr1":  self._get(r, "teff_err1"),
            "st_tefferr2":  self._get(r, "teff_err2"),

            "st_logg":      self._get(r, "logg"),
            "st_loggerr1":  self._get(r, "logg_err1"),
            "st_loggerr2":  self._get(r, "logg_err2"),

            "st_rad":       self._get(r, "radius"),
            "st_raderr1":   self._get(r, "radius_err1"),
            "st_raderr2":   self._get(r, "radius_err2"),

            "st_dist":      self._get(r, "dist"),
            "st_disterr1":  self._get(r, "dist_err1"),
            "st_disterr2":  self._get(r, "dist_err2"),

            "st_kepmag":    self._get(r, "kepmag"),

            "st_pmra":      self._get(r, "pmra"),
            "st_pmraerr":   None,  # pmra_err is not available in the NASA Exoplanet Archive
            "st_pmdec":     self._get(r, "pmdec"),
            "st_pmdecerr":  None,  # pmdec_err is not available in the NASA Exoplanet Archive
        }

        return {"stellar": st, "source_row": raw}

    # ----------------------- Small utilities ----------------------

    @staticmethod
    def _target_name(mission: str, target_id: Union[str, int]) -> str:
        tid = str(target_id).strip()
        if mission == "tess":
            return tid if tid.lower().startswith("tic") else f"TIC {tid}"
        return tid if tid.lower().startswith("kic") else f"KIC {tid}"

    @staticmethod
    def _get(row, name: str) -> Optional[Any]:
        if name not in row.colnames or row[name] is None:
            return None
        
        val = row[name]
        
        # Import necessary Astropy classes
        from astropy.coordinates import SkyCoord
        from astropy.units import Quantity
        
        # Handle SkyCoord objects which can't be simply converted with .item()
        if isinstance(val, SkyCoord):
            # Return the value in degrees
            return val.to_string("decimal")
        
        # Handle Quantity objects
        if isinstance(val, Quantity):
            # Return the value without the unit
            return val.value
        
        # Handle masked arrays
        if hasattr(val, "mask") and np.all(getattr(val, "mask")):
            return None
            
        # Regular case: extract scalar value
        try:
            return val.item()
        except (AttributeError, ValueError):
            # If .item() doesn't work, return as is
            return val

    @staticmethod
    def _as_float(val) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Astropy Row -> plain dict of ALL columns (JSON-friendly)."""
        out: Dict[str, Any] = {}
        
        # Import necessary Astropy classes
        from astropy.coordinates import SkyCoord
        from astropy.units import Quantity
        
        for name in row.colnames:
            val = row[name]
            
            # Handle SkyCoord objects
            if isinstance(val, SkyCoord):
                # Convert to string representation
                out[name] = val.to_string("decimal")
                continue
            
            # Handle Quantity objects
            if isinstance(val, Quantity):
                # Store the value without the unit
                out[name] = val.value
                continue
                
            # Handle masked arrays
            if hasattr(val, "mask"):
                if np.all(getattr(val, "mask")):
                    out[name] = None
                    continue
                val = val.filled(np.nan)
                
            # Try to convert to scalar values
            try:
                val = val.item()
            except Exception:
                pass
                
            # Handle NaN and Inf
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                val = None
                
            out[name] = val
        return out

    @staticmethod
    def _parse_int_id(val: Union[str, int], prefix: str) -> int:
        s = str(val).upper().replace(prefix.upper(), "").strip()
        return int(s)
