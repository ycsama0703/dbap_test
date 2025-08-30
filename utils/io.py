import pandas as pd
import os
import json

from pathlib import Path



'''
name could be: ['banks.json','households.json','insurance_companies.json','investment_advisors.json','mutual_funds.json','pension_funds.json','other.json']
'''




def json_data_to_df(name: str, align: str = "quarter_end") -> pd.DataFrame:
    """
    读取 data/<name>.json，为后续预测与评估统一时间列。
    - 将 fdate 解析为 datetime
    - 将 fdate 统一对齐到季度末(默认)或季度首
    - 派生:
        target_q    : 当前行所属季度 (Period['Q'])
        tplus1_q    : 下一季度
        tplus1_date : 下一季度的标准化日期（按 align 选择的首/末日）
    """

    THIS = Path(__file__).resolve()
    ROOT = THIS.parents[1]
    data_name = f"{name}.json" if not str(name).endswith(".json") else name
    path = ROOT / "data" / data_name

    df = pd.read_json(path, orient="records")


    s_num = pd.to_numeric(df["fdate"], errors="coerce")
    if s_num.notna().mean() > 0.8:

        unit = "ms" if s_num.dropna().median() > 1e11 else "s"
        df["fdate"] = pd.to_datetime(s_num, unit=unit, errors="coerce")
    else:

        df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")


    if align == "quarter_end":
        df["fdate"] = (
            df["fdate"].dt.to_period("Q").dt.asfreq("Q", "end").dt.to_timestamp()
        )
    elif align == "quarter_start":
        df["fdate"] = (
            df["fdate"].dt.to_period("Q").dt.asfreq("Q", "start").dt.to_timestamp()
        )
    else:
        raise ValueError("align must be 'quarter_end' or 'quarter_start'")


    df["target_q"] = df["fdate"].dt.to_period("Q")      
    df["tplus1_q"] = df["target_q"] + 1                 

    freq_side = "end" if align == "quarter_end" else "start"
    df["tplus1_date"] = df["tplus1_q"].dt.asfreq("Q", freq_side).dt.to_timestamp()


    key_cols = [c for c in ["permno", "mgrno", "fdate"] if c in df.columns]
    if key_cols:
        df = (
            df.sort_values(key_cols)
              .drop_duplicates(subset=key_cols, keep="first")
              .reset_index(drop=True)
        )
    return df




if __name__ == "__main__":
    print("Testing io.py ...")
    df = json_data_to_df("banks.json")
    print(df.head(3))
    print("Done !")