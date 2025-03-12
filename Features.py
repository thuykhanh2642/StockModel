"""Author: Kyle Nguyen"""
import pandas as pd
import numpy as np


def Technical_Indicators(data: pd.DataFrame) -> pd.DataFrame:
    tech_features = pd.DataFrame(index=data.index)
    
    """tech_features['SMA_5'] = data['Close'].rolling(window=5, min_periods=1).mean()
    tech_features['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()"""
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7, min_periods=1).mean()
    rs = gain / loss
    tech_features['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    tech_features['MACD'] = ema12 - ema26
    tech_features['MACD_signal'] = tech_features['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    """rolling_mean = data['Close'].rolling(window=10, min_periods=1).mean()
    rolling_std = data['Close'].rolling(window=10, min_periods=1).std()
    tech_features['BB_width'] = (rolling_std * 2) / rolling_mean
    return tech_features"""



def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    #MA
    features['ma_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    features['ma_50'] = data['Close'].rolling(window=60, min_periods=1).mean()
    
    #EMA
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    
        
    #Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    #Bollinger Band
    rolling_std = data['Close'].rolling(window=15, min_periods=1).std()
    features['bollinger_upper'] = features['ma_20'] + (2 * rolling_std)
    features['bollinger_lower'] = features['ma_20'] - (2 * rolling_std)
        
    for lag in [1, 5, 10, 15]:
        features[f'lag_{lag}'] = data['Close'].shift(lag)
    

    features.dropna(inplace=True)
    
    
    
    return features


