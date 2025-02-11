import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDataPreparer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with feature-engineered dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with all technical indicators
        """
        self.df = df.copy()
        
    def create_target(self, forecast_horizon: int = 1) -> None:
        """
        Create target variable for prediction.
        
        Args:
            forecast_horizon (int): Number of days ahead to predict
        """
        # Future close price
        self.df['future_close'] = self.df['close'].shift(-forecast_horizon)
        
        # Percentage change target
        self.df['target_return'] = (
            (self.df['future_close'] - self.df['close']) / self.df['close']
        ) * 100
        
        # Binary target (1 if price goes up)
        self.df['target_direction'] = (self.df['target_return'] > 0).astype(int)
        
        logger.info(f"Created target variables for {forecast_horizon} day(s) ahead")

    def select_features(self) -> List[str]:
        """
        Select relevant features for modeling.
        """
        # Features to exclude
        exclude_columns = [
            'date', 'future_close', 'target_return', 'target_direction',
            'dividends', 'stock splits'
        ]
        
        # Get feature columns
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        return feature_columns

    def prepare_sequences(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction.
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequences) and y (targets)
        """
        feature_columns = self.select_features()
        
        sequences = []
        targets = []
        
        for i in range(len(self.df) - sequence_length - 1):
            # Extract sequence
            sequence = self.df[feature_columns].iloc[i:(i + sequence_length)].values
            
            # Extract target
            target = self.df['target_direction'].iloc[i + sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    # Load feature-engineered data
    input_path = Path("data/processed/AAPL_features.csv")
    df = pd.read_csv(input_path)
    
    # Initialize preparer
    preparer = ModelDataPreparer(df)
    
    # Create target variables
    preparer.create_target(forecast_horizon=1)
    
    # Get feature columns
    feature_columns = preparer.select_features()
    
    # Prepare sequences
    X, y = preparer.prepare_sequences(sequence_length=10)
    
    # Print information
    print("\nFeature columns:", feature_columns)
    print("\nSequence shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Calculate class distribution
    class_distribution = np.mean(y) * 100
    print(f"\nPercentage of 'up' days: {class_distribution:.2f}%")
    
    # Save prepared data
    output_path = Path("data/processed/AAPL_model_ready.csv")
    preparer.df.dropna().to_csv(output_path, index=False)