from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from utils.pred_with_history import pipeline
from utils.api import get_response
from utils.io import json_data_to_df
from zoneinfo import ZoneInfo
from datetime import datetime

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


df2010 = pd.read_parquet(data_2010_path)
df2011 = pd.read_parquet(data_2011_path)
df2012 = pd.read_parquet(data_2012_path)
df2013 = pd.read_parquet(data_2013_path)
df2014 = pd.read_parquet(data_2014_path)
df2015 = pd.read_parquet(data_2015_path)
df2016 = pd.read_parquet(data_2016_path)
df2017 = pd.read_parquet(data_2017_path)
df2018 = pd.read_parquet(data_2018_path)
df2019 = pd.read_parquet(data_2019_path)

df = pd.concat([df2010, df2011,df2012,df2013,df2014,df2015,df2016,df2017,df2018,df2019], ignore_index = True)

cols = ['permno', 'fdate', 'type', 'me', 'be', 'profit', 'Gat', 'beta', 'holding','mgrno','mgrid']
sub_df = df[cols].copy()
os.makedirs("json_data", exist_ok=True)

for inv_type, group in sub_df.groupby("type"):
    
    filename = inv_type.replace(" ", "_").lower() + ".json"
    filepath = os.path.join("json_data", filename)
    
    
    group.to_json(filepath, orient="records", lines=False, force_ascii=False, indent=2)
    print(f"✅ Save {inv_type} to {filepath}")