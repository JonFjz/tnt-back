# mapper.py
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import math

# --------------------------
# Data classes
# --------------------------
@dataclass
class TOIDataRow:
    pl_trandurherr1: Optional[float]
    eng_transit_probability: Optional[float]
    pl_trandurherr2: Optional[float]
    eng_prad_srad_ratio: Optional[float]
    pl_orbpererr1: Optional[float]
    pl_orbper: Optional[float]
    eng_period_duration_ratio: Optional[float]
    eng_duration_period_ratio: Optional[float]
    pl_tranmiderr1: Optional[float]
    pl_trandeperr2: Optional[float]
    pl_tranmid: Optional[float]
    pl_trandep: Optional[float]
    pl_trandurh: Optional[float]
    pl_trandeperr1: Optional[float]
    st_tmagerr2: Optional[float]
    st_disterr2: Optional[float]
    st_loggerr2: Optional[float]
    st_disterr1: Optional[float]
    st_dist: Optional[float]
    st_teff: Optional[float]
    st_tmagerr1: Optional[float]
    st_rad: Optional[float]
    st_tefferr2: Optional[float]

@dataclass
class ModelPayload:
    model_type: str
    optimization_type: str
    model_name: str
    data: List[TOIDataRow] = field(default_factory=list)

    def to_dict(self):
        """Return a Python dictionary representation that can be directly used with Flask."""
        # Convert dataclass instances to dictionaries and ensure numeric types
        data_dicts = []
        for row in self.data:
            row_dict = asdict(row)
            # Convert None to 0 for numeric fields to avoid type issues
            for key, value in row_dict.items():
                if value is None:
                    row_dict[key] = 0.0
            data_dicts.append(row_dict)
            
        return {
            "model_type": self.model_type,
            "optimization_type": self.optimization_type,
            "model_name": self.model_name,
            "data": data_dicts,
        }
        
    def to_json(self):
        """Return a JSON string. Consider using to_dict() with Flask instead."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

# --------------------------
# Helpers
# --------------------------
def _f(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _abs_or_none(x: Optional[float]) -> Optional[float]:
    return None if x is None else abs(x)

def _compute_prad_srad_ratio(pl_rade: Optional[float], st_rad_sun: Optional[float]) -> Optional[float]:
    if pl_rade is None or st_rad_sun is None or st_rad_sun == 0:
        return None
    # 1 solar radius ≈ 109 Earth radii
    return pl_rade / (st_rad_sun * 109.0)

def _compute_period_duration_ratios(
    pl_orbper_days: Optional[float],
    pl_trandur_hours: Optional[float]
) -> tuple[Optional[float], Optional[float]]:

    # convert duration hours -> days so ratios are unitless
    if pl_orbper_days is None or pl_trandur_hours is None or pl_trandur_hours <= 0:
        return None, None
    dur_days = pl_trandur_hours / 24.0
    pd = pl_orbper_days / dur_days
    dp = dur_days / pl_orbper_days
    return pd, dp

def _estimate_transit_probability(row: Dict[str, Any]) -> Optional[float]:
    """
    Rough p_transit ≈ R_star / a, with:
      a = [G M (P*86400)^2 / (4π^2)]^(1/3)
      M from log g: g = 10^logg (cgs), M = g R^2 / G
    Needs st_logg, st_rad, pl_orbper. Returns None if missing.
    """
    pl_orbper = _f(row, "pl_orbper")
    st_logg = _f(row, "st_logg")
    st_rad = _f(row, "st_rad")
    if pl_orbper is None or st_logg is None or st_rad is None:
        return None
    try:
        G = 6.67430e-8          # cgs
        R_SUN = 6.957e10        # cm
        AU = 1.495978707e13     # cm
        P = pl_orbper * 86400.0
        g = 10.0 ** st_logg     # cm s^-2
        R = st_rad * R_SUN      # cm
        M = g * R * R / G       # g
        a_cm = (G * M * P * P / (4.0 * math.pi * math.pi)) ** (1.0 / 3.0)
        a_AU = a_cm / AU
        R_AU = R / AU
        prob = R_AU / a_AU
        # clamp into [0,1] just in case
        return max(0.0, min(1.0, prob))
    except Exception:
        return None

# Kepler (KOI) -> TOI-like field adapter
def _adapt_koi_to_toi_shape(row: Dict[str, Any]) -> Dict[str, Any]:
    # Map KOI columns to the TOI-like keys used below.
    m = dict(row)  # shallow copy
    def g(k): return row.get(k)

    # Period & duration & depth
    if "pl_orbper" not in m: m["pl_orbper"] = g("koi_period")
    if "pl_orbpererr1" not in m: m["pl_orbpererr1"] = g("koi_period_err1")
    if "pl_trandurh" not in m: m["pl_trandurh"] = g("koi_duration")
    if "pl_trandurherr1" not in m: m["pl_trandurherr1"] = g("koi_duration_err1")
    if "pl_trandurherr2" not in m: m["pl_trandurherr2"] = g("koi_duration_err2")
    if "pl_trandep" not in m: m["pl_trandep"] = g("koi_depth")
    if "pl_trandeperr1" not in m: m["pl_trandeperr1"] = g("koi_depth_err1")
    if "pl_trandeperr2" not in m: m["pl_trandeperr2"] = g("koi_depth_err2")

    # Mid-transit time: KOI uses BJD-2454833 (koi_time0bk). Convert to BJD if present.
    if "pl_tranmid" not in m:
        t0 = g("koi_time0bk")
        m["pl_tranmid"] = (t0 + 2454833.0) if t0 is not None else None
    if "pl_tranmiderr1" not in m:
        m["pl_tranmiderr1"] = g("koi_time0bk_err1")

    # Stellar params
    if "st_teff" not in m: m["st_teff"] = g("koi_steff")
    if "st_tefferr2" not in m: m["st_tefferr2"] = g("koi_steff_err2")
    if "st_logg" not in m: m["st_logg"] = g("koi_slogg")
    if "st_loggerr2" not in m: m["st_loggerr2"] = g("koi_slogg_err2")
    if "st_rad" not in m: m["st_rad"] = g("koi_srad")

    # Magnitude errors and distance often missing in KOI; leave None.
    if "st_tmagerr1" not in m: m["st_tmagerr1"] = None
    if "st_tmagerr2" not in m: m["st_tmagerr2"] = None
    if "st_dist" not in m: m["st_dist"] = None
    if "st_disterr1" not in m: m["st_disterr1"] = None
    if "st_disterr2" not in m: m["st_disterr2"] = None

    # Planet/Star radii for engineered ratio
    if "pl_rade" not in m: m["pl_rade"] = g("koi_prad")
    return m

# --------------------------
# Main mapper
# --------------------------
def build_model_payload_from_row(
    mission: str,
    row: Dict[str, Any],
    optimization_type: str = "recall",
    model_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ModelPayload:
    mission_lc = (mission or "").strip().lower()
    is_tess = mission_lc == "tess"
    is_kepler = mission_lc == "kepler"

    if not (is_tess or is_kepler):
        raise ValueError("mission must be 'tess' or 'kepler'")

    # Normalize row keys to TOI-like surface
    norm = row if is_tess else _adapt_koi_to_toi_shape(row)

    # Engineered features
    eng_prad_srad_ratio = _compute_prad_srad_ratio(_f(norm, "pl_rade"), _f(norm, "st_rad"))
    pd_ratio, dp_ratio = _compute_period_duration_ratios(_f(norm, "pl_orbper"), _f(norm, "pl_trandurh"))
    eng_transit_probability = _estimate_transit_probability(norm)

    # Apply overrides if provided (e.g., from ?parameters=...)
    overrides = overrides or {}
    if "eng_prad_srad_ratio" in overrides: eng_prad_srad_ratio = overrides["eng_prad_srad_ratio"]
    if "eng_period_duration_ratio" in overrides: pd_ratio = overrides["eng_period_duration_ratio"]
    if "eng_duration_period_ratio" in overrides: dp_ratio = overrides["eng_duration_period_ratio"]
    if "eng_transit_probability" in overrides: eng_transit_probability = overrides["eng_transit_probability"]

    # Build the row object (take absolute for *_err2 if negatives are stored)
    row_obj = TOIDataRow(
        pl_trandurherr1=_abs_or_none(_f(norm, "pl_trandurherr1")),
        eng_transit_probability=eng_transit_probability,
        pl_trandurherr2=_abs_or_none(_f(norm, "pl_trandurherr2")),
        eng_prad_srad_ratio=eng_prad_srad_ratio,
        pl_orbpererr1=_abs_or_none(_f(norm, "pl_orbpererr1")),
        pl_orbper=_f(norm, "pl_orbper"),
        eng_period_duration_ratio=pd_ratio,
        eng_duration_period_ratio=dp_ratio,
        pl_tranmiderr1=_abs_or_none(_f(norm, "pl_tranmiderr1")),
        pl_trandeperr2=_abs_or_none(_f(norm, "pl_trandeperr2")),
        pl_tranmid=_f(norm, "pl_tranmid"),
        pl_trandep=_f(norm, "pl_trandep"),
        pl_trandurh=_f(norm, "pl_trandurh"),
        pl_trandeperr1=_abs_or_none(_f(norm, "pl_trandeperr1")),
        st_tmagerr2=_abs_or_none(_f(norm, "st_tmagerr2")),
        st_disterr2=_abs_or_none(_f(norm, "st_disterr2")),
        st_loggerr2=_abs_or_none(_f(norm, "st_loggerr2")),
        st_disterr1=_abs_or_none(_f(norm, "st_disterr1")),
        st_dist=_f(norm, "st_dist"),
        st_teff=_f(norm, "st_teff"),
        st_tmagerr1=_abs_or_none(_f(norm, "st_tmagerr1")),
        st_rad=_f(norm, "st_rad"),
        st_tefferr2=_abs_or_none(_f(norm, "st_tefferr2")),
    )

    # Top-level model metadata
    model_type = "TOI" if is_tess else "KOI"
    if model_name is None:
        # Your default; feel free to branch by model_type if you want different names.
        model_name = "toi_recall_model_20251005"

    payload = ModelPayload(
        model_type=model_type,
        optimization_type=optimization_type,
        model_name=model_name,
        data=[row_obj],
    )
    return payload
