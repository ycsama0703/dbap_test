import pandas as pd
import os
import json
import re

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]   

def load_all_raw_parquet(folder="data/raw_data", columns=None, engine="pyarrow"):
    base = Path(folder)
    if not base.is_absolute():
        base = PROJECT_ROOT / base   

    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in: {base}")

    dfs = [pd.read_parquet(f, columns=columns, engine=engine) for f in files]
    return pd.concat(dfs, ignore_index=True, sort=False)

def save_json_by_type(df: pd.DataFrame, cols: list[str], chunk_size: int = 200_000) -> None:
    
    cols = list(cols)
    if 'type' not in cols: cols.append('type')
    keep = [c for c in cols if c in df.columns]
    sub = df.loc[:, keep].copy()

    
    if 'fdate' in sub.columns:
        sub['fdate'] = pd.to_datetime(sub['fdate'], errors='coerce').dt.strftime('%Y-%m-%d')

    outdir = PROJECT_ROOT / 'data' / 'json_data'
    outdir.mkdir(parents=True, exist_ok=True)

    for t, g in sub.groupby('type', dropna=False, observed=False):
        name = 'unknown' if pd.isna(t) else str(t).strip().replace(' ', '_').lower()
        path = outdir / f'{name}.json'

        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('[')
            first = True
            for i in range(0, len(g), chunk_size):
                part = g.iloc[i:i+chunk_size]
                
                s = part.to_json(orient='records', force_ascii=False)
                
                s = s[1:-1].strip()
                if s:
                    if not first:
                        f.write(',')
                    f.write(s)
                    first = False
            f.write(']')
        print(f"✅ saved {len(g):,} rows -> {path}")


def _resolve_json_path(name: str) -> Path:
    
    data_name = f"{name}.json" if not str(name).endswith(".json") else str(name)
    p = Path(data_name)
    data_dir = ROOT / "data"

    
    if p.parts and p.parts[0] == "json_data":
        cand = data_dir / p
        if cand.exists():
            return cand

    
    cand1 = data_dir / "json_data" / p.name
    if cand1.exists():
        return cand1

    
    cand2 = data_dir / p.name
    if cand2.exists():
        return cand2

    
    return cand1


def json_data_to_df(name: str, align: str = "quarter_end") -> pd.DataFrame:


    path = _resolve_json_path(name)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found. Tried: {path} and {ROOT/'data'/path.name}")

    df = pd.read_json(path, orient="records")


    s_num = pd.to_numeric(df["fdate"], errors="coerce")
    if s_num.notna().mean() > 0.8:

        unit = "ms" if s_num.dropna().median() > 1e11 else "s"
        df["fdate"] = pd.to_datetime(s_num, unit=unit, errors="coerce")
    else:

        df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")


    if align == "quarter_end":
        df["fdate"] = (
            df["fdate"].dt.to_period("Q").dt.asfreq("Q", "end").dt.to_timestamp()
        )
    elif align == "quarter_start":
        df["fdate"] = (
            df["fdate"].dt.to_period("Q").dt.asfreq("Q", "start").dt.to_timestamp()
        )
    else:
        raise ValueError("align must be 'quarter_end' or 'quarter_start'")


    df["target_q"] = df["fdate"].dt.to_period("Q")      
    df["tplus1_q"] = df["target_q"] + 1                 

    freq_side = "end" if align == "quarter_end" else "start"
    df["tplus1_date"] = df["tplus1_q"].dt.asfreq("Q", freq_side).dt.to_timestamp()


    key_cols = [c for c in ["permno", "mgrno", "fdate"] if c in df.columns]
    if key_cols:
        df = (
            df.sort_values(key_cols)
              .drop_duplicates(subset=key_cols, keep="first")
              .reset_index(drop=True)
        )
    return df


if __name__ == "__main__":

    ### Turn the raw data into json data by type.

    raw_dir = PROJECT_ROOT / "data" / "raw_data"
    print(f"[load] from: {raw_dir.resolve()}")
    df = load_all_raw_parquet(raw_dir)
    
    ###Select the col that you will use.
    cols = ['permno','fdate','type','me','be','profit','Gat','beta','holding','mgrno','aum','shares','prc'] 
    save_json_by_type(df, cols)
