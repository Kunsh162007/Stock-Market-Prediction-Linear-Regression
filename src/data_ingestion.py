import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, start_date, end_date, output_path=None):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {symbol}.")
            return None
        
        # Flatten MultiIndex columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data.reset_index(inplace=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
            
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    symbol = "^GSPC" # S&P 500
    start = "2010-01-01"
    end = "2024-01-01"
    output = "data/sp500_data.csv"
    fetch_stock_data(symbol, start, end, output)
