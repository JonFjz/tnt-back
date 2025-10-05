import json
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

MISSION_TO_MODEL = {"tess": "TOI", "kepler": "KOI"}

def normalize_items(items_like: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - a JSON string containing an array of objects, or
      - a Python list[dict].
    Returns list[dict] or raises ValueError.
    """
    if isinstance(items_like, str):
        items = json.loads(items_like)
    else:
        items = items_like
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise ValueError("items must be a list of objects or a JSON string of that.")
    return items

def build_payload(
    items_like: Any,
    optimization_type: str = "recall",
    model_name: Optional[str] = None,
    *,
    strict_mission: bool = True,
    allowed_features: Optional[Iterable[str]] = None,
    date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build:
    {
      "model_type": "TOI" | "KOI",
      "optimization_type": "<arg>",
      "model_name": "<auto or provided>",
      "data": [ {feature dicts...} ]
    }

    Rules:
      - Use `data` array if present and non-empty; otherwise fallback to `parameters`.
      - If `strict_mission`, all items must share the same mission.
      - If `allowed_features` provided, filter each feature dict to that set.
    """
    items = normalize_items(items_like)

    missions = [str(it.get("mission", "")).strip().lower() for it in items if it.get("mission")]
    if not missions:
        raise ValueError("Missing 'mission' in items.")
    if strict_mission and len(set(missions)) != 1:
        raise ValueError(f"Mixed missions with strict_mission=True: {set(missions)}")

    mission = missions[0]
    if mission not in MISSION_TO_MODEL:
        raise ValueError(f"Unsupported mission '{mission}'. Use 'tess' or 'kepler'.")

    model_type = MISSION_TO_MODEL[mission]
    opt = (optimization_type or "recall").strip().lower()

    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")
    if not model_name:
        model_name = f"{model_type.lower()}_{opt}_model_{date_str}"

    out_data: List[Dict[str, Any]] = []
    for it in items:
        blocks = it.get("data")
        if isinstance(blocks, list) and blocks:
            candidates = [b for b in blocks if isinstance(b, dict)]
        else:
            params = it.get("parameters")
            candidates = [params] if isinstance(params, dict) and params else []

        for feat in candidates:
            if allowed_features is not None:
                feat = {k: v for k, v in feat.items() if k in allowed_features}
            out_data.append(feat)

    if not out_data:
        raise ValueError("No feature dictionaries found under 'data' or 'parameters'.")

    return {
        "model_type": model_type,
        "optimization_type": opt,
        "model_name": model_name,
        "data": out_data,
    }