import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_stock_data(symbol: str, start_date: str, end_date: str, output_path: Path) -> pd.DataFrame:
    """
    Download stock data for a given symbol and date range.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_path (Path): Path to save the downloaded data
        
    Returns:
        pd.DataFrame: Downloaded stock data
    """
    try:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Download the data
        df = ticker.history(start=start_date, end=end_date)
        
        # Basic cleaning
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    start_date = "2018-01-01"
    end_date = "2025-01-01"
    output_path = Path("data/raw/AAPL_data.csv")
    
    df = download_stock_data(symbol, start_date, end_date, output_path)
    print(df.head())
    print("\nDataset shape:", df.shape)