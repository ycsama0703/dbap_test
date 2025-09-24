from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def save_json_by_type(df: pd.DataFrame, cols: list[str], partition_by_year=True) -> None:
    outdir = ROOT / "data" / "json_data"
    outdir.mkdir(parents=True, exist_ok=True)

    sub = df.loc[:, [c for c in cols if c in df.columns]].copy()
    sub["fdate"] = pd.to_datetime(sub["fdate"], errors="coerce", utc=True)

    for t, g in sub.groupby("type", dropna=False, observed=False):
        tname = "unknown" if pd.isna(t) else str(t).strip().replace(" ", "_").lower()
        if partition_by_year:
            for y, gy in g.groupby(g["fdate"].dt.year):
                path = outdir / f"{tname}_{y}.json"
                gy.to_json(path, orient="records", force_ascii=False)
                print(f"✅ {len(gy):,} rows -> {path}")
        else:
            path = outdir / f"{tname}.json"
            g.to_json(path, orient="records", force_ascii=False)
            print(f"✅ {len(g):,} rows -> {path}")

def save_parquet(df: pd.DataFrame, rel_path: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)