# Asset_Pricing/utils/build_single_prompt.py
from __future__ import annotations

from typing import Optional, Tuple, Union
import pandas as pd


__all__ = ["build_single_prompt"]


def _fmt_date(x) -> str:
    """Format anything to 'YYYY-MM-DD' (best effort)."""
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def build_single_prompt(
    row_t: pd.Series,
    prev_row: pd.Series,
    investor_role: str,
    investor_id: Union[int, str],
    hist_guard: Optional[Tuple[float, float]] = None,
) -> str:
    """
    预测 t+1 的持仓增量（Δ），仅输出 JSON: {"delta_holding": <float>}；Δ 可为负。
    输入： (t-2,t-1) 因子（t-2可缺省→用 NA），(t-1,t) 的真实持仓（锚）。
    """

    # ---------- helpers ----------
    def _fmt_date(x):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
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

    def _first_non_na(series: pd.Series, names):
        for n in names:
            if n in series.index:
                v = series[n]
                try:
                    if pd.isna(v):
                        continue
                except Exception:
                    pass
                return v
        return None

    # ---------- minimal validation ----------
    need_t   = ["permno","fdate","me","be","profit","Gat","beta","holding"]
    need_tm1 = ["permno","fdate","me","be","profit","Gat","beta"]
    miss_t   = [c for c in need_t   if c not in row_t.index]
    miss_tm1 = [c for c in need_tm1 if c not in prev_row.index]
    if miss_t:   raise KeyError(f"[build_single_prompt] row_t missing: {miss_t}")
    if miss_tm1: raise KeyError(f"[build_single_prompt] prev_row missing: {miss_tm1}")

    # ---------- timeline ----------
    stock_id  = row_t["permno"]
    fdate_t   = _fmt_date(row_t["fdate"])             # t
    fdate_tm1 = _fmt_date(prev_row["fdate"])          # t-1
    fdate_tm2 = _fmt_date(pd.to_datetime(prev_row["fdate"]) - pd.DateOffset(months=3))
    fdate_tp1 = _fmt_date(pd.to_datetime(row_t["fdate"]) + pd.DateOffset(months=3))

    # ---------- anchors (t-1, t) ----------
    holding_t   = row_t.get("holding", None)
    holding_tm1 = prev_row.get("holding", None)

    # ---------- fundamentals ----------
    me_tm1, be_tm1, profit_tm1, Gat_tm1, beta_tm1 = prev_row[["me","be","profit","Gat","beta"]]

    tm2_aliases = {
        "me":     ["me_tm2","me_lag2","me_t-2","me_t_minus_2"],
        "be":     ["be_tm2","be_lag2","be_t-2","be_t_minus_2"],
        "profit": ["profit_tm2","profit_lag2","profit_t-2","profit_t_minus_2"],
        "Gat":    ["Gat_tm2","Gat_lag2","Gat_t-2","Gat_t_minus_2"],
        "beta":   ["beta_tm2","beta_lag2","beta_t-2","beta_t_minus_2"],
    }
    me_tm2     = _first_non_na(row_t, tm2_aliases["me"])     or _first_non_na(prev_row, tm2_aliases["me"])
    be_tm2     = _first_non_na(row_t, tm2_aliases["be"])     or _first_non_na(prev_row, tm2_aliases["be"])
    profit_tm2 = _first_non_na(row_t, tm2_aliases["profit"]) or _first_non_na(prev_row, tm2_aliases["profit"])
    Gat_tm2    = _first_non_na(row_t, tm2_aliases["Gat"])    or _first_non_na(prev_row, tm2_aliases["Gat"])
    beta_tm2   = _first_non_na(row_t, tm2_aliases["beta"])   or _first_non_na(prev_row, tm2_aliases["beta"])

    # ---------- deltas (t-2→t-1) ----------
    def _delta(a,b):
        try:
            if (a is None) or (b is None) or pd.isna(a) or pd.isna(b): return None
            return float(a) - float(b)
        except Exception:
            return None

    d_me     = _delta(me_tm1,     me_tm2)
    d_be     = _delta(be_tm1,     be_tm2)
    d_profit = _delta(profit_tm1, profit_tm2)
    d_Gat    = _delta(Gat_tm1,    Gat_tm2)
    d_beta   = _delta(beta_tm1,   beta_tm2)

    # ---------- optional z-deltas & signal strength ----------
    z_aliases = {
        "profit": ["z_d_profit","zDelta_profit","z_delta_profit","profit_zdiff","z_profit_change"],
        "beta":   ["z_d_beta","zDelta_beta","z_delta_beta","beta_zdiff","z_beta_change"],
        "Gat":    ["z_d_Gat","zDelta_Gat","z_delta_Gat","Gat_zdiff","z_Gat_change"],
        "me":     ["z_d_me","zDelta_me","z_delta_me","me_zdiff","z_me_change"],
        "be":     ["z_d_be","zDelta_be","z_delta_be","be_zdiff","z_be_change"],
    }
    z_d_profit = _first_non_na(row_t, z_aliases["profit"]) or _first_non_na(prev_row, z_aliases["profit"])
    z_d_beta   = _first_non_na(row_t, z_aliases["beta"])   or _first_non_na(prev_row, z_aliases["beta"])
    z_d_Gat    = _first_non_na(row_t, z_aliases["Gat"])    or _first_non_na(prev_row, z_aliases["Gat"])
    z_d_me     = _first_non_na(row_t, z_aliases["me"])     or _first_non_na(prev_row, z_aliases["me"])
    z_d_be     = _first_non_na(row_t, z_aliases["be"])     or _first_non_na(prev_row, z_aliases["be"])

    try:
        z_terms = [z_d_profit, z_d_beta, z_d_Gat, z_d_me, z_d_be]
        if any((t is not None) and (not pd.isna(t)) for t in z_terms):
            S = sum(abs(float(t)) for t in z_terms if (t is not None) and (not pd.isna(t)))
        else:
            S = None
    except Exception:
        S = None

    # ---------- guard text ----------
    guard_text = ""
    if isinstance(hist_guard, tuple) and len(hist_guard) == 2:
        lo, hi = hist_guard
        try:
            guard_text = f"Historical scale hint: typical holding in [{lo:.4g}, {hi:.4g}] (not strict)"
        except Exception:
            guard_text = ""

    # ---------- reaction floor numbers ----------
    base_val = None
    shock_val = None
    try:
        if (holding_t is not None) and (holding_tm1 is not None) and not pd.isna(holding_t) and not pd.isna(holding_tm1):
            base_val = 0.10 * abs(float(holding_t) - float(holding_tm1))
        if (holding_t is not None) and not pd.isna(holding_t):
            ht = abs(float(holding_t))
            dp = abs(float(d_profit)) if d_profit is not None else 0.0
            db = abs(float(d_beta))   if d_beta   is not None else 0.0
            shock_val = 0.05 * (dp * ht) + 0.05 * (db * ht)
    except Exception:
        pass

    # ---------- blocks ----------
    t2_block = (
        "Fundamentals (t-2): "
        f"me={_fmt_val(me_tm2)}, be={_fmt_val(be_tm2)}, profit={_fmt_val(profit_tm2)}, "
        f"Gat={_fmt_val(Gat_tm2)}, beta={_fmt_val(beta_tm2)}"
    )
    t1_block = (
        "Fundamentals (t-1): "
        f"me={_fmt_val(me_tm1)}, be={_fmt_val(be_tm1)}, profit={_fmt_val(profit_tm1)}, "
        f"Gat={_fmt_val(Gat_tm1)}, beta={_fmt_val(beta_tm1)}"
    )
    delta_block = (
        "Changes (t-2→t-1): "
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
    base_txt  = _fmt_val(base_val)
    shock_txt = _fmt_val(shock_val)

    # ---------- prompt ----------
    prompt = f"""
Act as a quantitative portfolio manager at a {investor_role} institution.

Goal
- Predict the **increment** for next quarter (t+1) holding: `delta_holding = holding_(t+1) - holding_(t)`.
- Output **valid JSON only** with a single field `delta_holding` (it can be negative). No explanation.

Investor
- investor_id (mgrno): {investor_id}

Stock
- permno: {stock_id}

Timeline
- (t-2): {fdate_tm2}
- (t-1): {fdate_tm1}
- (t):   {fdate_t}
- (t+1): {fdate_tp1}

Recent realized holdings (same units)
- holding_(t-1) [{fdate_tm1}]: {_fmt_val(holding_tm1)}
- holding_(t)   [{fdate_t}]:   {_fmt_val(holding_t)}

{t2_block}
{t1_block}
{delta_block}
{z_block}
{S_block}

Constraints & Guidance
{guard_line}- **Do NOT copy anchors**: your `delta_holding` must not be ~0 unless changes justify it.
- **Direction rules**:
  - If Δprofit > 0 or β decreases → `delta_holding` should be **positive**.
  - If Δprofit < 0 or β increases → `delta_holding` should be **negative**.
- **Reaction floor (accelerate on shocks)**:
  - base = 0.10 * |holding_(t) - holding_(t-1)| → {base_txt}
  - shock = 0.05 * |Δprofit| * |holding_(t)| + 0.05 * |Δbeta| * |holding_(t)| → {shock_txt}
  - If S is provided and S ≥ 1.0, require `|delta_holding| ≥ base + shock`.
- **Bounds**:
  - Final holding will be computed as `max(0, holding_(t) + delta_holding)`; design `delta_holding` accordingly.

OUTPUT (valid JSON ONLY):
{{"delta_holding": <float>}}
""".strip()

    return prompt





if __name__ == "__main__":

    import pandas as pd


    row_t = pd.Series({
        "permno": 10107,
        "fdate":  "2012-06-30",   # 字符串日期最稳
        "me":     1.20,
        "be":     0.80,
        "profit": 0.15,
        "Gat":    0.05,
        "beta":   0.90,
        "holding": 1200.0,
    })

    prev_row = pd.Series({
        "permno": 10107,
        "fdate":  "2012-03-31",
        "me":     1.10,
        "be":     0.75,
        "profit": 0.12,
        "Gat":    0.04,
        "beta":   1.00,
    })

    try:
        prompt = build_single_prompt(
            row_t=row_t,
            prev_row=prev_row,
            investor_role="Banks",
            investor_id=7800,
            hist_guard=(100.0, 5000.0),  
        )
        print("\n==== Generated Prompt ====\n")
        print(prompt)

        assert "Current-quarter realized holding (t): 1200.0" in prompt
        assert '"holding_value": <value>' in prompt
        print("\n[OK] Basic checks passed.")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
