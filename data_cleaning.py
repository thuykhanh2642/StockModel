import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

def clean_data(df):
    if not isinstance(df.iloc[0, 0], pd.Timestamp):
        df = df.iloc[1:].reset_index(drop = True)

    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')

    columns = ['Close', 'High', 'Low','Open']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    df = df.sort_values('Date')
    df = df.ffill()  
    
    df = df.dropna()
    
    return df
