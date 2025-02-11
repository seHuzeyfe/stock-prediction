import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self, input_path: Path):
        """
        Initialize the data processor.
        
        Args:
            input_path (Path): Path to the raw data CSV file
        """
        self.input_path = input_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the raw data from CSV."""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Perform initial data preprocessing:
        1. Convert date to datetime
        2. Handle missing values
        3. Sort by date
        4. Calculate various return metrics
        """
        logger.info("Starting data preprocessing")
        
        # Convert date to datetime and handle timezone
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True).dt.tz_localize(None)
        
        # Sort by date
        self.df = self.df.sort_values('date')
        
        # Calculate different types of returns and movements
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['intraday_return'] = (self.df['close'] - self.df['open']) / self.df['open']
        
        # Trading signals
        self.df['intraday_up'] = (self.df['close'] > self.df['open']).astype(int)
        self.df['daily_up'] = (self.df['daily_return'] > 0).astype(int)
        
        # Calculate price ranges
        self.df['daily_range'] = (self.df['high'] - self.df['low']) / self.df['open']
        self.df['gap_up'] = (self.df['open'] > self.df['close'].shift(1)).astype(int)
        
        # Add day of week (0 = Monday, 4 = Friday)
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Handle missing values
        self.df = self.df.dropna()
        
        logger.info("Data preprocessing completed")
        return self.df

    def create_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.
        The split is time-based (not random) as this is time series data.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
        """
        split_idx = int(len(self.df) * (1 - test_size))
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        logger.info(f"Data split into train ({len(train_df)} rows) and test ({len(test_df)} rows)")
        return train_df, test_df

    def analyze_patterns(self) -> dict:
        """
        Analyze basic patterns in the data.
        """
        analysis = {
            'total_days': len(self.df),
            'intraday_up_percentage': self.df['intraday_up'].mean() * 100,
            'daily_up_percentage': self.df['daily_up'].mean() * 100,
            'avg_daily_return': self.df['daily_return'].mean() * 100,
            'avg_intraday_return': self.df['intraday_return'].mean() * 100,
            'avg_daily_range': self.df['daily_range'].mean() * 100
        }
        return analysis

    def save_processed_data(self, output_path: Path) -> None:
        """Save the processed data to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_path = Path("data/raw/AAPL_data.csv")
    output_path = Path("data/processed/AAPL_processed.csv")
    
    # Initialize and run processing
    processor = StockDataProcessor(input_path)
    processor.load_data()
    processor.preprocess_data()
    
    # Analyze patterns
    analysis = processor.analyze_patterns()
    print("\nPattern Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value:.2f}")
    
    # Create train/test split
    train_df, test_df = processor.create_train_test_split()
    
    # Save processed data
    processor.save_processed_data(output_path)
    
    # Print sample of processed data
    print("\nSample of processed data:")
    print(processor.df.head())
    print("\nDataset shape:", processor.df.shape)
    print("\nColumns:", processor.df.columns.tolist())