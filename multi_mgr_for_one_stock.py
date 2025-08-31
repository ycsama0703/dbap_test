# multi_mgr_for_one_stock.py
from pathlib import Path
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# —— 复用你项目内的工具（按你的相对路径）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import json_data_to_df
from utils.prompt_builder import build_single_prompt
from utils.api import get_response
from datetime import datetime, timedelta

# ---------- 小工具 ----------

def _standardize_quarter_start(dt_ser: pd.Series) -> pd.Series:
    """把日期标准化到季度首，便于对齐绘图/聚合。"""
    s = pd.to_datetime(dt_ser, errors="coerce")
    return s.dt.to_period("Q").dt.asfreq("Q", "start").dt.to_timestamp()

def eligible_mgrnos_for_permno(
    df: pd.DataFrame,
    permno: int,
    start_date: str = None,
    end_date: str = None,
    min_pairs: int = 1,
) -> list[int]:
    """
    找到在给定时间窗内、对该 permno 至少有 (t-1, t) 一对样本的 mgrno（即 len>=2）。
    """
    d = df[df["permno"] == permno].copy()
    d["fdate"] = pd.to_datetime(d["fdate"], errors="coerce")
    if start_date:
        d = d[d["fdate"] >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d["fdate"] <= pd.to_datetime(end_date)]

    # 每个 mgrno 在该股票上的样本数
    cnt = (d.drop_duplicates(["mgrno","fdate"])
             .groupby("mgrno")["fdate"].nunique())
    elig = cnt[cnt >= (min_pairs + 1)].index.tolist()  # 至少 2 个季度 → 1 对 (t-1,t)
    return sorted(elig)

def generate_prompts_list_for_mgr_stock(df, mgrno, permno, start_date, end_date, investor_role):
    """复用你单经理版本的逻辑，给某个 (mgrno, permno) 生成全时段 prompts。"""
    d = df[(df["mgrno"] == mgrno) & (df["permno"] == permno)].copy()
    d = d.sort_values("fdate").drop_duplicates(subset=["permno","mgrno","fdate"], keep="first")
    need = ["permno","fdate","me","be","profit","Gat","beta","mgrno","holding"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss} for mgrno={mgrno}, permno={permno}")

    # 过滤时间
    d["fdate"] = pd.to_datetime(d["fdate"], errors="coerce")
    if start_date:
        d = d[d["fdate"] >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d["fdate"] <= pd.to_datetime(end_date)]
    d = d.reset_index(drop=True)

    results = []
    for i in range(1, len(d)):
        prev_row = d.iloc[i-1]     # t-1
        row_t    = d.iloc[i]       # t
        prompt   = build_single_prompt(row_t=row_t, prev_row=prev_row,
                                       investor_role=investor_role, investor_id=mgrno)
        target_date = pd.to_datetime(row_t["fdate"]) + pd.offsets.QuarterEnd(1)  # t+1 的自然季度末
        results.append({
            "mgrno": mgrno,
            "permno": permno,
            "input_date": row_t["fdate"],
            "target_date": target_date,
            "prompt": prompt
        })
    return results

def call_model_and_collect(prompts, call_fn=get_response):
    """批量调用模型，返回 DataFrame：mgrno, permno, target_date, y_pred。"""
    from json import loads
    import re, time

    def extract_json_object(text):
        text = str(text).strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return m.group(0) if m else None

    rows = []
    for item in tqdm(prompts, desc="Querying model"):
        raw = call_fn(item["prompt"])
        js = extract_json_object(raw)
        y = np.nan
        if js:
            try:
                y = float(loads(js).get("holding_value", np.nan))
            except Exception:
                y = np.nan
        rows.append({
            "mgrno": item["mgrno"],
            "permno": item["permno"],
            "target_date": pd.to_datetime(item["target_date"]),
            "y_pred": y
        })
        time.sleep(0.1)  # 轻微节流
    dfp = pd.DataFrame(rows)
    # 标准化到季度首，便于横向对齐
    dfp["target_date_std"] = _standardize_quarter_start(dfp["target_date"])
    return dfp

def build_truth_for_permno(df, permno):
    """构造某股票在各 mgrno 下的真实 t 持仓（按季度首对齐），用于与预测的 t+1 对比或汇总。"""
    d = df[df["permno"] == permno].copy()
    d["fdate_std"] = _standardize_quarter_start(d["fdate"])
    truth = (d.groupby(["mgrno","permno","fdate_std"], as_index=False)["holding"]
               .sum()
               .rename(columns={"fdate_std":"target_date_std", "holding":"y_true"}))
    return truth

# ---------- 主流程：对一个 permno，遍历所有 mgrno ----------

def predict_for_stock_all_mgrs(
    df,
    permno: int,
    start_date: str,
    end_date: str,
    call_fn=get_response,
    investor_role="Banks",
    max_mgr: int | None = None,
    plot_lines: bool = True,
    plot_agg: bool = True,
    save_dir=None
):
    # 1) 找到合格的 mgrno（至少能形成一对 (t-1,t)）
    mgr_list = eligible_mgrnos_for_permno(df, permno, start_date, end_date, min_pairs=1)
    if max_mgr is not None:
        mgr_list = mgr_list[:max_mgr]
    if not mgr_list:
        raise RuntimeError(f"No eligible managers found for permno={permno} in {start_date}~{end_date}.")

    # 2) 为每个 mgr 生成 prompts 并调用模型
    all_prompts = []
    for m in mgr_list:
        ps = generate_prompts_list_for_mgr_stock(df, m, permno, start_date, end_date, investor_role)
        if ps:
            all_prompts.extend(ps)
    if not all_prompts:
        raise RuntimeError("No prompts generated. Check date window and data coverage.")

    preds = call_model_and_collect(all_prompts, call_fn=call_fn)

    # 3) 真值（同季度首对齐）
    truth = build_truth_for_permno(df, permno)

    # 4) 每个经理多条线
    if plot_lines:
        piv = preds.pivot_table(index="target_date_std", columns="mgrno", values="y_pred", aggfunc="mean")
        fig_lines, ax = plt.subplots(figsize=(10,5))
        piv.plot(ax=ax, marker="o", legend=True)
        ax.set_title(f"Predicted holdings for permno={permno} by manager (t+1)")
        ax.set_xlabel("Quarter"); ax.set_ylabel("Predicted holding")
        ax.grid(True)
        fig_lines.tight_layout()

    # 5) 汇总对比
    if plot_agg:
        agg_pred = preds.groupby(["permno","target_date_std"], as_index=False)["y_pred"].sum().rename(columns={"y_pred":"Hsum_pred"})
        agg_true = truth.groupby(["permno","target_date_std"], as_index=False)["y_true"].sum().rename(columns={"y_true":"Hsum_true"})
        agg = pd.merge(agg_pred, agg_true, on=["permno","target_date_std"], how="outer").sort_values("target_date_std")

        fig_agg, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(agg["target_date_std"], agg["Hsum_true"], marker="o", label="True total holding")
        ax2.plot(agg["target_date_std"], agg["Hsum_pred"], marker="o", label="Pred total holding (sum over mgr)")
        ax2.set_title(f"Total holdings for permno={permno}: predicted vs true")
        ax2.set_xlabel("Quarter"); ax2.set_ylabel("Total holding")
        ax2.grid(True); ax2.legend(); fig_agg.tight_layout()

    # === 保存 ===
    if save_dir:
        outdir = Path(save_dir)
    else:
        # 默认存到项目根的 outputs/ 下
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(__file__).resolve().parents[1] / "outputs" / f"permno_{permno}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    if plot_lines:
        fig_lines.savefig(outdir / f"mgr_lines_perm{permno}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_lines)
    if plot_agg:
        fig_agg.savefig(outdir / f"agg_perm{permno}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_agg)

    # 也把数据存一下，方便复用
    preds.to_csv(outdir / "preds_mgr_perm.csv", index=False)
    truth.to_csv(outdir / "truth_mgr_perm.csv", index=False)
    print(f"Saved figures and CSVs to: {outdir}")

    return preds, truth, mgr_list

# ---------- 直接运行示例 ----------
if __name__ == "__main__":
    inv_type = "banks"
    df = json_data_to_df(inv_type + ".json")  # 你已有的读取函数，会标准化时间列
    permno = 10107
    start_date = "2010-01-01"
    end_date   = "2019-12-31"

    preds, truth, mgrs = predict_for_stock_all_mgrs(
        df, permno=10107, start_date="2010-01-01", end_date="2019-12-31",
        call_fn=get_response, investor_role="Banks",
        save_dir=None,             # 留空=自动存到 outputs/
        plot_lines=True, plot_agg=True,max_mgr = 10
    )
    print(f"[Done] managers used: {len(mgrs)} → {mgrs[:10]}{' ...' if len(mgrs)>10 else ''}")
