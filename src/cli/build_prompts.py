from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.hfcore.prompts.builder import build_prompt

ROOT = Path(__file__).resolve().parents[2]

def build_jsonl(df: pd.DataFrame, out_path: str | Path, investor_role_col="type"):
    out = []
    df = df.sort_values(["mgrno","permno","date"])
    # 找到每个 (mgr, permno) 的 t 与 t-1
    g = df.groupby(["mgrno","permno"], sort=False)
    for (_, _), grp in g:
        if len(grp) < 2: 
            continue
        for i in range(1, len(grp)):
            row_t   = grp.iloc[i]
            prevrow = grp.iloc[i-1]
            prompt = build_prompt(
                row_t=row_t,
                prev_row=prevrow,
                investor_role=str(row_t.get(investor_role_col, "Institution")),
                investor_id=row_t["mgrno"],
                hist_guard=None
            )
            out.append({
                "mgrno": int(row_t["mgrno"]),
                "permno": int(row_t["permno"]),
                "date": pd.to_datetime(row_t["date"]).strftime("%Y-%m-%d"),
                "prompt": prompt,
                "target_holding": float(row_t["holding_t1"]),  # 供训练/评估
            })
    op = ROOT / "artifacts" / "prompts" / "train.jsonl"
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        for r in out:
            f.write(pd.io.json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ saved {len(out)} prompts -> {op}")

if __name__ == "__main__":
    # 假设你把 processed 面板另存为 data/processed/panel_quarter.parquet
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel_quarter.parquet")
    build_jsonl(df, ROOT / "artifacts" / "prompts" / "train.jsonl")