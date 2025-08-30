from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from utils.pred_with_history import pipeline
from utils.api import get_response
from utils.io import json_data_to_df
from zoneinfo import ZoneInfo

import json
import time
import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def pretty_role(s: str) -> str:
    return s.replace("_", " ").title()

if __name__ == "__main__":
    # 1) 角色只影响 prompt；数据固定用 Banks.json
    inv_type_list = [
        "banks",
        "households",
        "insurance_companies",
        "investment_advisors",
        "mutual_funds",
        "pension_funds",
    ]

    # 2) 读取数据（固定银行数据）
    data_name = "Banks.json"
    df = json_data_to_df(data_name)

    # 3) 固定一个投资者/股票与时间窗
    mgrno = 7800
    permno = 10107
    start_date = "2012-04-01"
    end_date = "2013-07-01"

    # 4) 逐角色跑，并收集结果
    combined = None
    rows_for_metrics = []
    pred_cols = []

    for role in inv_type_list:
        role_for_prompt = pretty_role(role)  # 仅用于 prompt 与图例
        eval_df, metrics, preds, prompts = pipeline(
            df,
            mgrno,
            permno,
            start_date,
            end_date,
            get_response,
            role_for_prompt,
            plot=False,   # 单次不画图，最后统一画
        )

        cur = eval_df[["target_date", "y_true", "y_pred"]].copy()
        pred_col = f"pred_{role}"
        cur.rename(columns={"y_pred": pred_col}, inplace=True)
        pred_cols.append(pred_col)

        if combined is None:
            combined = cur
        else:
            combined = combined.merge(cur[["target_date", pred_col]], on="target_date", how="outer")

        rows_for_metrics.append({"role": role, **metrics})

    # 5) 统一画一张图
    combined = combined.sort_values("target_date").drop_duplicates(subset=["target_date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    if "y_true" in combined.columns:
        ax.plot(combined["target_date"], combined["y_true"], marker="o", linewidth=2, label="Actual")

    for role in inv_type_list:
        col = f"pred_{role}"
        if col in combined.columns:
            m = combined[col].notna()
            ax.plot(
                combined.loc[m, "target_date"],
                combined.loc[m, col],
                marker="o",
                linestyle="--",
                label=pretty_role(role),
            )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Holding")
    ax.set_title(f"Role Sweep (mgrno={mgrno}, permno={permno})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # 6) 计算并保存指标/图表
    metrics_df = pd.DataFrame(rows_for_metrics)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    outdir = Path("outputs") / f"role_sweep_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(outdir / "combined_predictions.csv", index=False)
    metrics_df.to_csv(outdir / "metrics_by_role.csv", index=False)
    fig.savefig(outdir / "role_sweep_plot.png", dpi=150, bbox_inches="tight")

    print("\n[Summary] Metrics by role:")
    print(metrics_df.assign(role=metrics_df["role"].map(pretty_role)))
    print(f"\nSaved to: {outdir}")