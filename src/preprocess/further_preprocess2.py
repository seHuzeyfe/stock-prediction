import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def remove_multicollinearity(self, threshold: float = 0.7, keep_important: List[str] = None) -> pd.DataFrame:
        """
        Remove highly correlated features while retaining essential ones.
        The non-feature columns (e.g., date) are excluded from the calculation.
        """
        if keep_important is None:
            # Ensure that essential features and the target ('close') are retained.
            keep_important = ['rsi', 'intraday_return', 'close']

        # We exclude 'date' from feature calculations.
        non_feature_cols = ['date']
        features_df = self.df.drop(columns=non_feature_cols, errors='ignore')
        corr_matrix = features_df.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Drop features that exceed the threshold and are not in the keep_important list.
        to_drop = [col for col in upper.columns if any(upper[col] > threshold) and col not in keep_important]
        logger.info(f"Dropping {len(to_drop)} features due to multicollinearity: {to_drop}")
        return self.df.drop(columns=to_drop)

    def select_features(self) -> List[str]:
        """
        Select relevant features based on availability.
        Updated to use the target 'close' for regression instead of 'target_direction'.
        """
        keep_features = [
            'open', 'close', 'volume', 'dividends', 'stock splits',
            'daily_return', 'intraday_return', 'daily_range', 'gap_up',
            'day_of_week', 'rsi', 'bb_width', 'macd', 'volume_ratio', 'target_return'
        ]
        
        # Ensure only existing columns are selected.
        available_features = [col for col in keep_features if col in self.df.columns]
        return available_features

class ModelDataPreparer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = RobustScaler()  # Using RobustScaler to mitigate the impact of outliers.
        
    def prepare_features(self) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Select and filter features.
        selector = FeatureSelector(self.df)
        self.df = selector.remove_multicollinearity()
        selected_features = selector.select_features()
        
        # Scale only the feature columns (excluding temporal column 'date').
        temporal_cols = ['date']
        feature_cols = [col for col in selected_features if col not in temporal_cols]
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        
        return self.df[selected_features + ['date']]  # Ensure 'date' is preserved.

    def prepare_sequences(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction.
        The target is the next day's closing price ('close').
        """
        features_df = self.prepare_features()
        
        # Ensure the 'date' column is set as index if needed.
        features_df = features_df.sort_values('date')
        features_df.reset_index(drop=True, inplace=True)
        
        # Define feature columns (all except 'date' and the target 'close').
        feature_columns = [col for col in features_df.columns if col not in ['date', 'close']]
        features = features_df[feature_columns].values
        
        # Define the target as the closing price.
        target_values = features_df['close'].values
        
        sequences = []
        targets_list = []
        
        # Build sequences such that each target is the close price following the sequence.
        for i in range(len(features_df) - sequence_length):
            sequence = features[i:(i + sequence_length)]
            target = target_values[i + sequence_length]
            sequences.append(sequence)
            targets_list.append(target)
            
        return np.array(sequences), np.array(targets_list)

if __name__ == "__main__":
    # Load data. Ensure that the CSV includes a 'close' column for the target.
    input_path = Path("data/processed/AAPL_model_ready.csv")
    df = pd.read_csv(input_path, parse_dates=['date'])
    
    # Prepare data.
    preparer = ModelDataPreparer(df)
    X, y = preparer.prepare_sequences(sequence_length=10)
    
    # Print information.
    print("\nSequence shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFeatures used:", [col for col in preparer.df.columns if col not in ['date'] and col != 'close'])
    
    # Print summary statistics for the target variable.
    print(f"\nTarget (closing price) statistics: min={np.min(y):.2f}, max={np.max(y):.2f}, mean={np.mean(y):.2f}")
    
    # Save prepared data.
    output_path = Path("data/processed/AAPL_model_ready_final.csv")
    preparer.df.to_csv(output_path, index=False)
