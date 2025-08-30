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


    # ---- Minimal, defensive input validation (fail-fast with clear messages) ----
    need_t = ["permno", "fdate", "me", "be", "profit", "Gat", "beta", "holding"]
    need_tm1 = ["permno", "fdate", "me", "be", "profit", "Gat", "beta"]

    missing_t = [c for c in need_t if c not in row_t.index]
    missing_tm1 = [c for c in need_tm1 if c not in prev_row.index]
    if missing_t:
        raise KeyError(f"[build_prompt_m3] row_t missing columns: {missing_t}")
    if missing_tm1:
        raise KeyError(f"[build_prompt_m3] prev_row missing columns: {missing_tm1}")

    # ---- Extract + format fields ----
    stock_id = row_t["permno"]
    fdate_t = _fmt_date(row_t["fdate"])
    fdate_tm1 = _fmt_date(prev_row["fdate"])
    fdate_tp1 = _fmt_date(pd.to_datetime(row_t["fdate"]) + pd.DateOffset(months=3))

    me_t, be_t, profit_t, Gat_t, beta_t = row_t[["me", "be", "profit", "Gat", "beta"]]
    me_tm1, be_tm1, profit_tm1, Gat_tm1, beta_tm1 = prev_row[
        ["me", "be", "profit", "Gat", "beta"]
    ]
    holding_t = row_t["holding"]

    guard_text = ""
    if isinstance(hist_guard, tuple) and len(hist_guard) == 2:
        lo, hi = hist_guard
        try:
            guard_text = (
                f"- Historical scale hint: typical holding in [{lo:.4g}, {hi:.4g}] (not strict)\n"
            )
        except Exception:
            guard_text = ""

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
