from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.backends.deepseek_api import DeepSeekBackend
from src.hfcore.utils.parsers import parse_holding

ROOT = Path(__file__).resolve().parents[2]

def run():
    # 读取 prompts jsonl
    inp = ROOT / "artifacts" / "prompts" / "train.jsonl"
    rows = []
    backend = DeepSeekBackend()   # 可切换 OpenAIBackend

    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            rec = pd.io.json.loads(line)
            prompt = rec["prompt"]
            raw_out = backend.generate(prompt)
            holding_pred = parse_holding(raw_out)
            rows.append({
                "mgrno": rec["mgrno"],
                "permno": rec["permno"],
                "date": rec["date"],
                "target_holding": rec["target_holding"],
                "raw_out": raw_out,
                "holding_pred": holding_pred,
            })

    out_df = pd.DataFrame(rows)
    op = ROOT / "predictions" / "latest" / "preds.parquet"
    op.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(op, index=False)
    print(f"✅ saved {len(out_df)} predictions -> {op}")

if __name__ == "__main__":
    run()