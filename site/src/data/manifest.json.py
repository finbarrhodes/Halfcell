"""Cache manifest (summaries, params, model metrics, feature importances)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
print(json.dumps(json.loads((ROOT / "data/cache/manifest.json").read_text())))
