
from pathlib import Path
import sys

if __package__ in (None, ""):
    
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]          
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Asset_Pricing.utils.api import get_response
    from Asset_Pricing.utils.io import json_data_to_df
    from Asset_Pricing.utils.prompt_builder import build_single_prompt
else:
    
    from .api import get_response
    from .io import json_data_to_df
    from .prompt_builder import build_single_prompt
# --- end dual-import header ---

from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo

import json
import time
import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def extract_json_object(text):
    """Try to extract a JSON object from free-form text.
    Returns the json string (e.g. '{"holding_value": 12.3}') or None.
    """
    # Common case: the model returns ONLY JSON
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    # Fallback: regex the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def parse_model_json_to_float(json_str):
    """Parse a json string like '{"holding_value": 12.3}' into a float.
    Returns float('nan') if parsing fails.
    """
    try:
        obj = json.loads(json_str)
        val = obj.get("holding_value", None)
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")

def prompts_to_preds_raw(prompts, call_fn=get_response, max_retries=3, backoff=1.5, sleep_between=0.2):
    """Loop over prompts list and call the model to build preds_raw.
    Each item in `prompts` is expected to be a dict with keys:
      - "target_date": pd.Timestamp (or str)
      - "prompt": str

    Returns: list of dicts: [{"target_date": <Timestamp>, "model_json": "<json str>"}]
    """
    preds_raw = []
    for item in tqdm(prompts, desc="Querying model", total=len(prompts)):
        prompt = item["prompt"]
        target_date = item["target_date"]
        # Ensure target_date is serializable
        if not isinstance(target_date, pd.Timestamp):
            try:
                target_date = pd.to_datetime(target_date)
            except Exception:
                pass

        # simple retry loop
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_text = call_fn(prompt)
                json_str = extract_json_object(raw_text)
                if json_str is None:
                    raise ValueError("No JSON object found in model response.")
                # Keep the raw JSON string; parsing happens later in evaluation
                preds_raw.append({
                    "target_date": target_date,
                    "model_json": json_str
                })
                break
            except Exception as e:
                last_err = e
                wait = (backoff ** (attempt - 1))
                print(f"[Warn] Model call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            # after loop without break
            print(f"[Error] Failed to get JSON after {max_retries} attempts. Skipping this item.")
            preds_raw.append({
                "target_date": target_date,
                "model_json": json.dumps({"holding_value": None})
            })

        # small courtesy sleep for rate limits
        time.sleep(sleep_between)

    print(f"[Info] Collected {len(preds_raw)} model responses.")
    return preds_raw


def build_prompt_m3(
    row_t, prev_row, investor_role, investor_id, hist_guard=None
):
    """Build a single prompt for one (permno, fdate) row with (t) anchor."""
    def _fmt(x):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return str(x)

    stock_id = row_t["permno"]
    fdate_t  = _fmt(row_t["fdate"])
    fdate_tm1 = _fmt(prev_row["fdate"])
    fdate_tp1 = _fmt(pd.to_datetime(row_t["fdate"]) + pd.DateOffset(months=3))

    # current-quarter features (t)
    me_t, be_t, profit_t, Gat_t, beta_t = row_t[["me","be","profit","Gat","beta"]]
    holding_t = row_t["holding"]  

    # previous-quarter features (t-1)
    me_tm1, be_tm1, profit_tm1, Gat_tm1, beta_tm1 = prev_row[["me","be","profit","Gat","beta"]]

    guard_text = ""
    if isinstance(hist_guard, tuple) and len(hist_guard) == 2:
        lo, hi = hist_guard
        guard_text = f"- Historical scale hint: typical holding in [{lo:.4g}, {hi:.4g}] (not strict)\n"

    prompt = f"""
Act as a quantitative portfolio manager at a {investor_role} institution.

Task: Predict the **next quarter (t+1)** holding (same unit as dataset `holding`) for a **specific investor** in a **specific stock**,
using previous-quarter fundamentals (t-1) and current-quarter fundamentals (t), with **current-quarter realized holding (t)** as the anchor.

Investor:
- investor_type: {investor_role}
- investor_id (mgrno): {investor_id}

Stock:
- stock_id (permno): {stock_id}

Timeline:
- previous_quarter (t-1): {fdate_tm1}
- current_quarter  (t):   {fdate_t}
- target_quarter   (t+1): {fdate_tp1}

Context / Anchors:
{guard_text}- Current-quarter realized holding (t): {holding_t}

Previous-quarter fundamentals (t-1):
me={me_tm1}, be={be_tm1}, profit={profit_tm1}, Gat={Gat_tm1}, beta={beta_tm1}

Current-quarter fundamentals (t):
me={me_t}, be={be_t}, profit={profit_t}, Gat={Gat_t}, beta={beta_t}

Guidance:
- Use **(t) realized holding** as the primary anchor and adjust toward (t) fundamentals and the change from (t-1) to (t).
- Prediction should NOT equal market equity.
- Prefer higher holdings for more profitable and lower-beta firms.
- Keep the prediction within a reasonable order-of-magnitude of **(t)**.
- Output must be a single non-negative float. No explanation.

OUTPUT (valid JSON ONLY):
{{"holding_value": <value>}}
"""
    return prompt



def generate_prompts_list(
    df: pd.DataFrame,
    mgrno: int,
    permno: int,
    start_date: str = None,
    end_date: str = None,
    investor_role: str = None
):
    """
    Generate prompts for all available input pairs (t-1, t) to predict holding at t+1,
    for a specific investor (mgrno) and stock (permno).
    """
    df = df.copy()

    # Filter to investor & stock
    df = df[(df["mgrno"] == mgrno) & (df["permno"] == permno)].copy()


    # Sort and deduplicate
    df = df.sort_values("fdate").drop_duplicates(subset=["permno","mgrno","fdate"], keep="first")

    # Check required columns
    needed_cols = ["permno","fdate","me","be","profit","Gat","beta","mgrno","holding"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []
    df = df.reset_index(drop=True)

    # Iterate using iloc so row_t / prev_row are Series (not dict)
    for i in tqdm(range(1, len(df)), desc="Generating M3 Prompts"):
        prev_row = df.iloc[i-1]  # t-1
        row_t    = df.iloc[i]    # t

        input_date  = row_t["fdate"]
        target_date = pd.to_datetime(row_t["fdate"]) + pd.offsets.QuarterEnd(1)

        prompt = build_single_prompt(
            row_t=row_t,
            prev_row=prev_row,
            investor_role=investor_role,
            investor_id=mgrno
        )

        results.append({
            "input_date": input_date,
            "target_date": target_date,
            "mgrno": mgrno,
            "permno": permno,
            "prompt": prompt
        })

    # print(f"[Info] Built {len(results)} prompts (with prev anchor) for mgrno={mgrno}, permno={permno}.")
    return results

def parse_model_output_to_float(text):
    """
    Parse a JSON string like '{"holding_value": 123.45}' and return the float.
    Return np.nan if parsing fails.
    """
    try:
        obj = json.loads(text)
        val = obj.get("holding_value", None)
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan

def build_prediction_df(preds_raw):
    """
    Build a tidy DataFrame from raw model outputs.
    Required keys in each dict: 'target_date', 'model_json'
    """
    df_pred = pd.DataFrame(preds_raw).copy()
    # ensure datetime
    df_pred["target_date"] = pd.to_datetime(df_pred["target_date"])
    # parse JSON to float
    df_pred["y_pred"] = df_pred["model_json"].apply(parse_model_output_to_float)
    # keep only necessary
    df_pred = df_pred[["target_date", "y_pred"]]
    return df_pred

def get_ground_truth(df, mgrno, permno):
    """
    Return a DataFrame with columns: ['target_date','y_true'] for the given (mgrno, permno).
    It uses df_banks['fdate'] as the 'target_date' and df_banks['holding'] as 'y_true'.
    """
    df = df.copy()

    # ensure datetime for fdate
    if not pd.api.types.is_datetime64_any_dtype(df["fdate"]):
        s_num = pd.to_numeric(df["fdate"], errors="coerce")
        unit = "ms" if s_num.dropna().median() > 1e11 else "s"
        df["fdate"] = pd.to_datetime(s_num, unit=unit, errors="coerce")

    df = df[(df["mgrno"] == mgrno) & (df["permno"] == permno)].copy()
    df = df[["fdate", "holding"]].rename(columns={"fdate": "target_date", "holding": "y_true"})
    # If duplicates exist at the same target_date, aggregate (sum) or choose first
    df = df.groupby("target_date", as_index=False)["y_true"].sum()
    return df

def evaluate_predictions(df, mgrno, permno, preds_raw, zero_eps=1e-9, plot=True):
    print("[Info] Building prediction DataFrame...")
    df_pred = build_prediction_df(preds_raw).copy()
    df_pred["target_date"] = pd.to_datetime(df_pred["target_date"], errors="coerce")
    df_pred["target_q"] = df_pred["target_date"].dt.to_period("Q")   # ★ 预测用季度标签

    print("[Info] Extracting ground truth...")
    df_true = get_ground_truth(df, mgrno=mgrno, permno=permno).copy()
    df_true["target_date"] = pd.to_datetime(df_true["target_date"], errors="coerce")
    df_true["target_q"] = df_true["target_date"].dt.to_period("Q")   # ★ 真值用季度标签

    print("[Info] Merging on target_q (quarter)...")
    eval_df = pd.merge(df_true, df_pred, on="target_q", how="inner").sort_values("target_q")

    if eval_df.empty:
        print("[Warn] No overlapping quarters after merge. Check your dates.")
        return eval_df, {"count": 0, "MAE": None, "RMSE": None, "MAPE_%": None}, None

    eval_df["target_date"] = eval_df["target_q"].dt.asfreq("Q", "start").dt.to_timestamp()

    eval_df["err"] = eval_df["y_pred"] - eval_df["y_true"]
    eval_df["abs_err"] = eval_df["err"].abs()
    eval_df["pct_err"] = eval_df["abs_err"] / (eval_df["y_true"].abs() + zero_eps)

    mae = eval_df["abs_err"].mean()
    rmse = math.sqrt((eval_df["err"]**2).mean())
    mape = (eval_df["pct_err"].mean()) * 100.0
    metrics = {"count": int(len(eval_df)), "MAE": mae, "RMSE": rmse, "MAPE_%": mape}

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(eval_df["target_date"], eval_df["y_true"], marker="o", label="Actual")
        ax.plot(eval_df["target_date"], eval_df["y_pred"], marker="o", label="Predicted")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Holding")
        ax.set_title(f"Investor {mgrno} - Stock {permno}: Predicted vs Actual")
        ax.grid(True); ax.legend(); fig.tight_layout()

    return eval_df, metrics, fig


def _jsonify(o):
    if isinstance(o, (pd.Timestamp, datetime)):
  
        try:
            if pd.isna(o):
                return None
        except Exception:
            pass
        return o.isoformat()

   
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(o, np.bool_):
        return bool(o)

  
    if isinstance(o, np.ndarray):
        return o.tolist()

   
    return str(o)

def pipeline(
    df,
    mgrno,
    permno,
    start_date,
    end_date,
    call_fn,
    investor_role,
    plot=True
):
    """
    Run Method-3 end-to-end: prompts (with prev anchor) -> model -> evaluation.
    Returns (eval_df, metrics, preds_raw, prompts).
    """



    print("[Stage] Generating prompts...")
    prompts = generate_prompts_list(
        df=df,
        mgrno=mgrno,
        permno=permno,
        start_date=start_date,
        end_date=end_date,
        investor_role=investor_role
    )

    print("[Stage] Calling model to obtain predictions...")
    preds_raw = prompts_to_preds_raw(prompts, call_fn=call_fn)


    print("[Stage] Evaluating predictions against ground truth...")
    eval_df, metrics,fig = evaluate_predictions(
        df=df,
        mgrno=mgrno,
        permno=permno,
        preds_raw=preds_raw,
        plot=plot
    )


    #save the output
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if fig is not None:
        figpath = output_dir / f"pred_vs_actual_{mgrno}_{permno}.png"
        fig.savefig(figpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    eval_df.to_csv(output_dir / "eval_df.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_jsonify)

    with open(output_dir / "preds_raw.json", "w", encoding="utf-8") as f:
        json.dump(preds_raw, f, indent=2, ensure_ascii=False, default=_jsonify)

    with open(output_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False, default=_jsonify)

    return eval_df, metrics, preds_raw, prompts




if __name__ == "__main__":

    inv_type = "Banks"
    data_name = inv_type + ".json"
    df = json_data_to_df(data_name)
    mgrno = 7800
    permno = 10107
    start_date = "2012-04-01"
    end_date = "2013-07-01"
    call_fn = get_response
    investor_role = inv_type

    
    eval_df, metrics, preds, prompts = pipeline(
        df,
        mgrno,
        permno,
        start_date,
        end_date,
        get_response,     # your API caller
        inv_type,
        plot=True
    )

    print(eval_df.head(5))
    print(metrics)
 
