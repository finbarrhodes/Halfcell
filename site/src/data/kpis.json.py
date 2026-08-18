"""Market snapshot KPIs for the landing page."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
print(json.dumps(json.loads((ROOT / "data/cache/latest_kpis.json").read_text())))
