import yfinance as yf
from datetime import datetime,timedelta
import time
import pandas as pd
import os
from data_cleaning import clean_data
import numpy as np

"""it will now try the API first, if fail then check for backup file """

def retrieveStock(stock, start_date="2017-01-01", end_date="2017-01-01"):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    backup_folder = "Backup"
    retries = 2
    backup_path = os.path.join(backup_folder, f"{stock}.csv")
    os.makedirs(backup_folder, exist_ok=True)

    for attempt in range(retries):
        try:
            print(f"Attempting to retrieve {stock} data from API (attempt {attempt + 1})...")
            stockHist = yf.download(stock, start=start_date, end=end_date)
            stockHist.reset_index(inplace=True) #resetting index

            #multi index fit
            if isinstance(stockHist.columns, pd.MultiIndex) and len(stockHist.columns.levels) > 1:
                #only drop if it has more than 1 level
                stockHist.columns = stockHist.columns.droplevel(0)
            if not pd.to_datetime(stockHist.iloc[0, 0], errors='coerce'):
                stockHist = stockHist.iloc[1:].reset_index(drop=True)

            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
            stockHist = stockHist.iloc[:, :len(expected_columns)]  # Keep only expected columns
            stockHist.columns = expected_columns[:len(stockHist.columns)]  # Assign correct names

            #Date to datetime
            stockHist['Date'] = pd.to_datetime(stockHist['Date'], errors='coerce')

            #numeric columns to float
            numeric_cols = ['Close', 'High', 'Low', 'Open']
            for col in numeric_cols:
                stockHist[col] = pd.to_numeric(stockHist[col], errors='coerce').astype(np.float64)


            #Save backup file
            if not stockHist.empty:
                print(f"Successfully retrieved data from API for {stock}")
                stockHist.to_csv(backup_path, index=False)
                print(f"Backup saved to {backup_path}")
                return stockHist

        except Exception as e:
            print(f"API attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print("All API attempts failed. Checking for backup file...")

    #if API fails
    if os.path.exists(backup_path):
        print(f"Loading data from backup file: {backup_path}")
        stockHist = pd.read_csv(backup_path, parse_dates=['Date'])
        stockHist = stockHist[(stockHist['Date'] >= start_date) & (stockHist['Date'] <= end_date)]

        if not stockHist.empty:
            print(f"Successfully loaded backup data for {stock}")
            return stockHist
        else:
            print("Backup file exists but contains no valid data.")
            return None

    print(f"No backup file found for {stock}.")
    return None
