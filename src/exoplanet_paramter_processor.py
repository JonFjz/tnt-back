# exoplanet_parameter_processor_min.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional, Set
import numpy as np

@dataclass
class ExoplanetParameterProcessorTEST:
    fits_path: Optional[str]     # can be None if you run catalog-only
    mission: str
    catalog: Dict[str, Any]      # includes st_* fields already
    overrides: Dict[str, Any]    # optional: period/duration/depth, etc.
    prefer_flux: str = "pdcsap"  # advisory for LCF selection
    required_fields: Set[str] = None

    def run(self) -> Dict[str, Any]:
        req = self.required_fields or set()

        # 1) Seed from overrides (fast path if TOI/KOI supplied)
        out: Dict[str, Any] = {}
        seed = {**(self.overrides or {})}

        # 2) If period/duration/depth present, skip transit search entirely
        have_core = all(k in seed for k in ("pl_orbper","pl_trandurh","pl_trandep"))
        if not have_core and self.fits_path:
            # Heavy path only if any needed field depends on the LC
            if any(k in req for k in ("eng_period_duration_ratio",
                                      "eng_duration_period_ratio",
                                      "eng_prad_srad_ratio",
                                      "eng_transit_probability")):
                # Load light curve and do the minimal computation
                # (placeholder; hook your existing detrend+BLS here if truly required)
                pass

        # 3) Fill engineered fields cheaply from what we have
        #    Only compute what was requested.
        st_rad = self._f(self.catalog, "st_rad")
        if "eng_prad_srad_ratio" in req:
            # If you have planet radius (koi_prad or pl_rade) use it; else compute from depth if needed
            prad = seed.get("koi_prad") or seed.get("pl_rade")
            if prad is None and "pl_trandep" in seed and st_rad:
                # transit depth δ ~ (Rp/Rs)^2  -> Rp/Rs = sqrt(δ), Rp = (Rp/Rs)*Rs
                ror = np.sqrt(max(seed["pl_trandep"], 0.0))
                prad = ror * st_rad
            out["eng_prad_srad_ratio"] = None if prad is None or st_rad in (None, 0) else float(prad)/float(st_rad)

        if "eng_period_duration_ratio" in req:
            P = seed.get("pl_orbper")
            D = seed.get("pl_trandurh")
            out["eng_period_duration_ratio"] = (None if (P is None or D is None or D == 0)
                                                else float(P) / float(D))

        if "eng_duration_period_ratio" in req:
            P = seed.get("pl_orbper")
            D = seed.get("pl_trandurh")
            out["eng_duration_period_ratio"] = (None if (P is None or P == 0 or D is None)
                                                else float(D) / float(P))

        if "eng_transit_probability" in req:
            # Simple geometric approximation: ~ (R*/a) for small impact parameters
            # Use period + stellar mass for a via Kepler's 3rd if needed. Keep it lightweight.
            out["eng_transit_probability"] = self._geometric_transit_prob(
                period_days=seed.get("pl_orbper"), st_rad=st_rad, st_mass=self._f(self.catalog, "st_mass")
            )

        # 4) Return only required fields + passthrough of known inputs
        passthrough = {k: seed.get(k) for k in ("pl_orbper","pl_trandurh","pl_trandep",
                                                "pl_orbpererr1","pl_trandurherr1","pl_trandurherr2",
                                                "pl_trandeperr1","pl_trandeperr2","pl_tranmid","pl_tranmiderr1")}
        out |= {k:v for k,v in passthrough.items() if k in req}
        # include stellar ones if requested
        for k in ("st_teff","st_tefferr1","st_tefferr2","st_logg","st_loggerr2",
                  "st_rad","st_raderr1","st_raderr2","st_dist","st_disterr1","st_disterr2",
                  "st_tmag","st_tmagerr1","st_tmagerr2"):
            if k in req:
                out[k] = self._f(self.catalog, k)

        return out

    @staticmethod
    def _f(d: Dict[str, Any], k: str):
        v = d.get(k)
        try:
            return float(v) if v is not None else None
        except Exception:
            return v

    @staticmethod
    def _geometric_transit_prob(period_days, st_rad, st_mass):
        if period_days is None or st_rad is None or st_mass is None:
            return None
        # a/Rsun ≈ ( (G*M_sun)*(P*86400)^2 / (4π^2) )^(1/3) / R_sun
        # Using constants folded into 215.032 * (M/Msun)^(1/3) * (P/365d)^(2/3)
        try:
            a_over_rsun = 215.032 * (float(st_mass))**(1/3) * (float(period_days)/365.25)**(2/3)
            # R*/a in solar units -> st_rad / a_over_rsun
            return max(min(float(st_rad) / a_over_rsun, 1.0), 0.0)
        except Exception:
            return None
