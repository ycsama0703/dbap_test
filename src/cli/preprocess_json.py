from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]  # 项目根：.../your-project/

RENAME_FACTORS = {
    "me": "factor1",
    "be": "factor2",
    "profit": "factor3",
    "Gat": "factor4",
    "beta": "factor5",
}

NEEDED_MIN = ["mgrno", "permno", "fdate", "holding", *RENAME_FACTORS.keys()]


def _read_json_any(path: Path) -> pd.DataFrame:
    """同时兼容 .json (数组) 与 .jsonl (一行一个对象)。"""
    txt = path.read_text(encoding="utf-8", errors="ignore").lstrip()
    if txt.startswith("["):
        # JSON 数组
        data = json.loads(txt)
        return pd.DataFrame(data)
    else:
        # JSONL
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)


def _quarter_align(s: pd.Series, where: str = "end") -> pd.Series:
    # where in {"end","start"}
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    q = dt.dt.to_period("Q")
    return q.dt.asfreq("Q", where).dt.to_timestamp(tz="UTC")


def preprocess(
    in_path: Path,
    out_path: Path,
    quarter_where: str = "end",
    dropna_label: bool = True,
) -> Path:
    df = _read_json_any(in_path)

    # --- 基础列检查 ---
    miss = [c for c in ["mgrno", "permno", "fdate", "holding"] if c not in df.columns]
    if miss:
        raise KeyError(f"缺少必要列：{miss}；现有列：{list(df.columns)[:20]} ...")

    # --- 重命名因子列（若存在）---
    for old, new in RENAME_FACTORS.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # --- 时间对齐到季度 ---
    df["date"] = _quarter_align(df["fdate"], where=quarter_where)

    # --- 排序去重 ---
    df = df.sort_values(["mgrno", "permno", "date"]).drop_duplicates(
        subset=["mgrno", "permno", "date"], keep="first"
    )

    # --- 生成 t 与 t+1 ---
    df["holding_t"] = df["holding"]
    df["holding_t1"] = df.groupby(["mgrno", "permno"], sort=False)["holding_t"].shift(
        -1
    )

    # --- 选择/重排列 ---
    keep_cols = [
        "mgrno",
        "permno",
        "date",
        "holding_t",
        "holding_t1",
        "factor1",
        "factor2",
        "factor3",
        "factor4",
        "factor5",
        "prc",
        "shares",
        "type",
        "aum",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[keep_cols].copy()

    # --- 可选：去掉没有标签的行（末期会没有 t+1）---
    if dropna_label:
        before = len(out)
        out = out.dropna(subset=["holding_t1"]).reset_index(drop=True)
        print(f"drop rows without label: {before - len(out)}")

    # --- 季度字符串（方便筛选/报表）---
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)

    # --- 保存 parquet ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"✅ saved {len(out):,} rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess JSON/JSONL to processed panel with holding_t1.")
    parser.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(ROOT / "data" / "raw" / "dataset.json"),
        help="输入 .json 或 .jsonl 路径",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        default=str(ROOT / "data" / "processed" / "panel_quarter.parquet"),
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--quarter",
        dest="quarter",
        choices=["end", "start"],
        default="end",
        help="季度对齐到季末(end)或季初(start)",
    )
    parser.add_argument(
        "--keep-nan-label",
        action="store_true",
        help="保留没有 holding_t1 的行（默认丢弃）",
    )
    args = parser.parse_args()

    preprocess(
        in_path=Path(args.inp),
        out_path=Path(args.out),
        quarter_where=args.quarter,
        dropna_label=not args.keep_nan_label,
    )


if __name__ == "__main__":
    main()