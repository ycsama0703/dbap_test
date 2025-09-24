from __future__ import annotations
from typing import Optional, Tuple, Union
import pandas as pd

__all__ = ["build_prompt"]

_FACT_ALIASES_T = ["me","be","profit","Gat","beta"]
_FACT_ALIASES_STD = ["factor1","factor2","factor3","factor4","factor5"]

def _fmt_date(x) -> str:
    try:
        return pd.to_datetime(x, utc=True).strftime("%Y-%m-%d")
    except Exception:
        return str(x)

def _fmt_val(v):
    if v is None:
        return "NA"
    try:
        if pd.isna(v):
            return "NA"
    except Exception:
        pass
    return f"{v}"

def _delta(a, b):
    try:
        if (a is None) or (b is None) or pd.isna(a) or pd.isna(b):
            return None
        return float(a) - float(b)
    except Exception:
        return None

def _pick5(series: pd.Series):
    """支持两套列名：me/be/profit/Gat/beta 或 factor1..factor5"""
    if set(_FACT_ALIASES_T).issubset(series.index):
        return series[_FACT_ALIASES_T].tolist()
    elif set(_FACT_ALIASES_STD).issubset(series.index):
        return series[_FACT_ALIASES_STD].tolist()
    else:
        raise KeyError("Expected either {me,be,profit,Gat,beta} or {factor1..factor5}")

def build_prompt(
    row_t: pd.Series,
    prev_row: pd.Series,
    investor_role: str,
    investor_id: Union[int, str],
    hist_guard: Optional[Tuple[float, float]] = None,
) -> str:
    """构造单样本 prompt；输出要求模型只返回 JSON：{"holding_tp1": <float>}"""

    need_t   = ["permno","fdate","holding"]
    need_tm1 = ["permno","fdate"]
    miss_t   = [c for c in need_t   if c not in row_t.index]
    miss_tm1 = [c for c in need_tm1 if c not in prev_row.index]
    if miss_t:   raise KeyError(f"[build_prompt] row_t missing: {miss_t}")
    if miss_tm1: raise KeyError(f"[build_prompt] prev_row missing: {miss_tm1}")

    stock_id  = row_t["permno"]
    fdate_t   = _fmt_date(row_t["fdate"])
    fdate_tm1 = _fmt_date(prev_row["fdate"])
    fdate_tp1 = _fmt_date(pd.to_datetime(row_t["fdate"], utc=True) + pd.DateOffset(months=3))

    holding_t   = row_t.get("holding", None)
    holding_tm1 = prev_row.get("holding", None)

    me_tm1, be_tm1, profit_tm1, Gat_tm1, beta_tm1 = _pick5(prev_row)
    me_t,   be_t,   profit_t,   Gat_t,   beta_t   = _pick5(row_t)

    d_me     = _delta(me_t,     me_tm1)
    d_be     = _delta(be_t,     be_tm1)
    d_profit = _delta(profit_t, profit_tm1)
    d_Gat    = _delta(Gat_t,    Gat_tm1)
    d_beta   = _delta(beta_t,   beta_tm1)

    # 可选 z 标准化变化：从两行里尽力找
    def _first_non_na(series: pd.Series, names):
        for n in names:
            if n in series.index:
                v = series[n]
                try:
                    if pd.isna(v): continue
                except Exception:
                    pass
                return v
        return None

    z_alias = {
        "profit": ["z_d_profit","zDelta_profit","z_delta_profit","profit_zdiff","z_profit_change"],
        "beta":   ["z_d_beta","zDelta_beta","z_delta_beta","beta_zdiff","z_beta_change"],
        "Gat":    ["z_d_Gat","zDelta_Gat","z_delta_Gat","Gat_zdiff","z_Gat_change"],
        "me":     ["z_d_me","zDelta_me","z_delta_me","me_zdiff","z_me_change"],
        "be":     ["z_d_be","zDelta_be","z_delta_be","be_zdiff","z_be_change"],
    }
    z_d_profit = _first_non_na(row_t, z_alias["profit"]) or _first_non_na(prev_row, z_alias["profit"])
    z_d_beta   = _first_non_na(row_t, z_alias["beta"])   or _first_non_na(prev_row, z_alias["beta"])
    z_d_Gat    = _first_non_na(row_t, z_alias["Gat"])    or _first_non_na(prev_row, z_alias["Gat"])
    z_d_me     = _first_non_na(row_t, z_alias["me"])     or _first_non_na(prev_row, z_alias["me"])
    z_d_be     = _first_non_na(row_t, z_alias["be"])     or _first_non_na(prev_row, z_alias["be"])

    try:
        z_terms = [z_d_profit, z_d_beta, z_d_Gat, z_d_me, z_d_be]
        S = sum(abs(float(t)) for t in z_terms if (t is not None) and (not pd.isna(t))) if any(
            (t is not None) and (not pd.isna(t)) for t in z_terms
        ) else None
    except Exception:
        S = None

    guard_text = ""
    if isinstance(hist_guard, tuple) and len(hist_guard) == 2:
        lo, hi = hist_guard
        try:
            guard_text = f"Historical scale hint: typical holding in [{lo:.4g}, {hi:.4g}] (not strict)"
        except Exception:
            guard_text = ""

    shock_txt = "NA"
    try:
        if (holding_t is not None) and not pd.isna(holding_t):
            ht = abs(float(holding_t))
            dp = abs(float(d_profit)) if d_profit is not None else 0.0
            db = abs(float(d_beta))   if d_beta   is not None else 0.0
            shock_txt = f"{0.05 * (dp * ht) + 0.05 * (db * ht):.6g}"
    except Exception:
        pass

    t1_block = (
        "Fundamentals (t-1): "
        f"me={_fmt_val(me_tm1)}, be={_fmt_val(be_tm1)}, profit={_fmt_val(profit_tm1)}, "
        f"Gat={_fmt_val(Gat_tm1)}, beta={_fmt_val(beta_tm1)}"
    )
    t_block = (
        "Fundamentals (t): "
        f"me={_fmt_val(me_t)}, be={_fmt_val(be_t)}, profit={_fmt_val(profit_t)}, "
        f"Gat={_fmt_val(Gat_t)}, beta={_fmt_val(beta_t)}"
    )
    delta_block = (
        "Changes (t-1→t): "
        f"Δme={_fmt_val(d_me)}, Δbe={_fmt_val(d_be)}, Δprofit={_fmt_val(d_profit)}, "
        f"ΔGat={_fmt_val(d_Gat)}, Δbeta={_fmt_val(d_beta)}"
    )
    z_block = (
        "Standardized changes (optional): "
        f"zΔprofit={_fmt_val(z_d_profit)}, zΔbeta={_fmt_val(z_d_beta)}, "
        f"zΔGat={_fmt_val(z_d_Gat)}, zΔme={_fmt_val(z_d_me)}, zΔbe={_fmt_val(z_d_be)}"
    )
    S_block = f"Signal strength S (optional): {_fmt_val(S)}"
    guard_line = f"- {guard_text}\n" if guard_text else ""

    prompt = f"""
Act as a quantitative portfolio manager at a {investor_role} institution.

Goal
- Predict the **next-quarter absolute holding** `holding_(t+1)`.
- Output **valid JSON only** with a single field `holding_tp1` (it can be 0). No explanation.

Investor
- investor_id (mgrno): {investor_id}

Stock
- permno: {stock_id}

Timeline
- (t-1): {fdate_tm1}
- (t):   {fdate_t}
- (t+1): {fdate_tp1}

Recent realized holdings (same units)
- holding_(t)   [{fdate_t}]:   {_fmt_val(holding_t)}{f"\\n- holding_(t-1) [{fdate_tm1}]: {_fmt_val(holding_tm1)}" if holding_tm1 is not None else ""}

{t1_block}
{t_block}
{delta_block}
{z_block}
{S_block}

Constraints & Guidance
{guard_line}- **Direction rules**:
  - If Δprofit > 0 or β decreases → `holding_(t+1)` should be **higher** than `holding_(t)`.
  - If Δprofit < 0 or β increases → `holding_(t+1)` should be **lower** than `holding_(t)`.
- **Magnitude hint (if S or shocks are present)**:
  - shock ≈ 0.05 * |Δprofit| * |holding_(t)| + 0.05 * |Δbeta| * |holding_(t)| → {shock_txt}
  - If S is provided and S ≥ 1.0, avoid tiny adjustments relative to the above shock hint.
- **Bounds**:
  - `holding_(t+1) ≥ 0` (non-negative).
  - Design a plausible value considering the historical scale hint if provided.

OUTPUT (valid JSON ONLY):
{{"holding_tp1": <float>}}
""".strip()
    return prompt