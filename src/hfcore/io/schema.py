from __future__ import annotations
import pandera as pa
from pandera.typing import Series
import pandas as pd

class RawPanel(pa.DataFrameModel):
    permno: Series[int] = pa.Field(coerce=True)
    mgrno:  Series[int] = pa.Field(coerce=True)
    fdate:  Series[object]  # 后面统一转
    type:   Series[str] | Series[object]
    holding: Series[float] = pa.Field(nullable=True)
    prc:     Series[float] = pa.Field(nullable=True)
    me:      Series[float] = pa.Field(nullable=True)
    be:      Series[float] = pa.Field(nullable=True)
    profit:  Series[float] = pa.Field(nullable=True)
    Gat:     Series[float] = pa.Field(nullable=True)
    beta:    Series[float] = pa.Field(nullable=True)
    aum:     Series[float] = pa.Field(nullable=True)
    shares:  Series[float] = pa.Field(nullable=True)

    class Config:
        strict = False  # 允许多余列

class ProcessedPanel(pa.DataFrameModel):
    permno: Series[int]
    mgrno: Series[int]
    date: Series[pd.Timestamp]
    quarter: Series[str]
    holding_t: Series[float]
    holding_t1: Series[float]    # 目标
    prc_t: Series[float]
    factor1: Series[float] | Series[object]
    factor2: Series[float] | Series[object]
    factor3: Series[float] | Series[object]
    factor4: Series[float] | Series[object]
    factor5: Series[float] | Series[object]