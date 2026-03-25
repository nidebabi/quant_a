from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent
PRESET_PATH = ROOT / "reports" / "factor_presets.json"


def _ensure_parent():
    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)


def _json_safe_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return str(v)


def _json_safe_dict(d: dict):
    return {str(k): _json_safe_value(v) for k, v in d.items()}


def load_factor_presets():
    _ensure_parent()
    if not PRESET_PATH.exists():
        return {}

    try:
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_factor_preset(name: str, config: dict):
    _ensure_parent()
    presets = load_factor_presets()
    presets[str(name)] = _json_safe_dict(config)
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)


def delete_factor_preset(name: str):
    _ensure_parent()
    presets = load_factor_presets()
    if name in presets:
        presets.pop(name)
        with open(PRESET_PATH, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)