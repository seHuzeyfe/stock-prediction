import pandas as pd
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a dataframe containing OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        
    def add_moving_averages(self, windows: List[int] = [5, 10, 20]) -> None:
        """Add Simple Moving Averages (SMA) for specified windows."""
        for window in windows:
            self.df[f'sma_{window}'] = self.df['close'].rolling(window=window).mean()
            # Add distance from price to MA as a percentage
            self.df[f'dist_to_sma_{window}'] = (
                (self.df['close'] - self.df[f'sma_{window}']) / self.df[f'sma_{window}']
            ) * 100

    def add_rsi(self, window: int = 14) -> None:
        """Add Relative Strength Index."""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2) -> None:
        """Add Bollinger Bands."""
        self.df['bb_middle'] = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()
        
        self.df['bb_upper'] = self.df['bb_middle'] + (std * num_std)
        self.df['bb_lower'] = self.df['bb_middle'] - (std * num_std)
        
        # Add BB width and position within bands
        self.df['bb_width'] = (
            (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        ) * 100
        
        self.df['bb_position'] = (
            (self.df['close'] - self.df['bb_lower']) / 
            (self.df['bb_upper'] - self.df['bb_lower'])
        )

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """Add Moving Average Convergence Divergence."""
        exp1 = self.df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = exp1 - exp2
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']

    def add_volume_indicators(self) -> None:
        """Add volume-based indicators."""
        # Volume Moving Average
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
        
        # Volume relative to moving average
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']

    def generate_all_features(self) -> pd.DataFrame:
        """Generate all technical indicators."""
        logger.info("Generating technical indicators...")
        
        self.add_moving_averages()
        self.add_rsi()
        self.add_bollinger_bands()
        self.add_macd()
        self.add_volume_indicators()
        
        # Drop any rows with NaN values resulting from calculations
        self.df = self.df.dropna()
        
        logger.info("Technical indicators generated successfully")
        return self.df

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Load processed data
    input_path = Path("data/processed/AAPL_processed.csv")
    df = pd.read_csv(input_path)
    
    # Generate features
    indicator_generator = TechnicalIndicators(df)
    df_with_features = indicator_generator.generate_all_features()
    
    # Save enhanced dataset
    output_path = Path("data/processed/AAPL_features.csv")
    df_with_features.to_csv(output_path, index=False)
    
    # Print summary
    print("\nFeatures generated:")
    print(df_with_features.columns.tolist())
    print("\nDataset shape:", df_with_features.shape)
    print("\nSample of generated features:")
    print(df_with_features[['close', 'rsi', 'macd', 'bb_position']].head())