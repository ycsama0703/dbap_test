from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schema import RawPanel

ROOT = Path(__file__).resolve().parents[3]  # 项目根

def load_all_raw_parquet(folder="data/raw", columns=None, engine="pyarrow") -> pd.DataFrame:
    base = (ROOT / folder) if not Path(folder).is_absolute() else Path(folder)
    files = sorted(base.glob("*.parquet")) + sorted(base.glob("*.parq"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {base}")
    df = pd.concat([pd.read_parquet(f, columns=columns, engine=engine) for f in files],
                   ignore_index=True, sort=False)
    return RawPanel.validate(df)

def read_json_dataset(name: str) -> pd.DataFrame:
    """
    读取 data/json_data/{name}.json 或 data/{name}.json
    """
    p = ROOT / "data" / "json_data" / f"{name}.json"
    if not p.exists():
        p = ROOT / "data" / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_json(p, orient="records")