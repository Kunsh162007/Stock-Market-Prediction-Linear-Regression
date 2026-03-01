import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col='Close', n_lags=5):
    """
    Cleans data and engineers features for Linear Regression.
    """
    df = df.copy()
    
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        
    # Feature Engineering: Lag features
    for i in range(1, n_lags + 1):
        df[f'Lag_{i}'] = df[target_col].shift(i)
        
    # Feature Engineering: Moving Averages
    df['SMA_20'] = df[target_col].rolling(window=20).mean()
    df['SMA_50'] = df[target_col].rolling(window=50).mean()
    
    # Target: Next day's Close price
    df['Target'] = df[target_col].shift(-1)
    
    # Drop NaNs created by lagging and shifting
    df.dropna(inplace=True)
    
    # Features and Target
    feature_cols = [f'Lag_{i}' for i in range(1, n_lags + 1)] + ['SMA_20', 'SMA_50']
    X = df[feature_cols]
    y = df['Target']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, df['Date'], scaler

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("data/sp500_data.csv")
    X, y, dates, scaler = preprocess_data(data)
    print("Features preview:")
    print(X.head())
    print(f"Shapes: X={X.shape}, y={y.shape}")
