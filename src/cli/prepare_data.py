from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.hfcore.io.readers import load_all_raw_parquet
from src.hfcore.io.writers import save_json_by_type, save_parquet
from src.hfcore.io.schema import ProcessedPanel

def quarter_align(s: pd.Series, where="end"):
    q = s.dt.to_period("Q")
    return q.dt.asfreq("Q", where).dt.to_timestamp(tz="UTC")

def run():
    # 1) 读 raw
    df = load_all_raw_parquet("data/raw")

    # 2) 可选：把 raw 导出为按 type/年分区的 JSON
    keep = ['permno','fdate','type','me','be','profit','Gat','beta','holding','mgrno','aum','shares','prc']
    save_json_by_type(df, keep)

    # 3) 生成 processed（季度对齐 + t+1 目标）
    df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce", utc=True)
    df["date"] = quarter_align(df["fdate"], where="end")
    df = df.sort_values(["mgrno","permno","date"])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # 目标：同 (mgrno, permno) 的下一季持仓
    df["holding_t1"] = df.groupby(["mgrno","permno"])["holding"].shift(-1)
    proc = df.rename(columns={
        "holding": "holding_t",
        "prc": "prc_t",
        "me": "factor1", "be": "factor2", "profit": "factor3", "Gat": "factor4", "beta": "factor5"
    })[
        ["permno","mgrno","date","quarter",
         "holding_t","holding_t1","prc_t","factor1","factor2","factor3","factor4","factor5"]
    ].dropna(subset=["holding_t","holding_t1"])

    proc = ProcessedPanel.validate(proc)
    save_parquet(proc, "data/processed/panel_quarter.parquet")
    print("✅ processed saved to data/processed/panel_quarter.parquet")

if __name__ == "__main__":
    run()