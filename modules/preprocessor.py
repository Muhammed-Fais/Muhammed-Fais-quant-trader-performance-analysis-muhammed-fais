import pandas as pd
import numpy as np
from helpers.logger import setup_logger

logger = setup_logger("Preprocessor")

class Preprocessor:
    def __init__(self):
        self.features = [
            'num_trades', 'total_profit_x', 'win_rate',
            'avg_profit_rate', 'std_profit_rate', 'total_commission',
            'total_lot_size', 'avg_lot_size', 'avg_duration_hr', 'profit_factor',
            'max_win_streak', 'max_loss_streak', 'risk_reward_ratio', 'inv_max_loss_streak'
        ]

    def preprocess_raw(self, raw_data: list) -> pd.DataFrame:
        """
        raw_data: list of dicts representing multiple trades with user_id, profit, commission, etc.

        Returns:
            pd.DataFrame with one row per user_id and aggregated features.
        """
        logger.info("Starting raw data preprocessing...")

        df = pd.DataFrame(raw_data)

        # Check expected columns are present
        expected_cols = ['user_id', 'profit', 'profit_rate', 'commission', 'lot_size', 'duration_hr']
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns in raw data: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        grouped = df.groupby('user_id')

        logger.info("Aggregating features per user_id...")

        agg_df = pd.DataFrame({
            'num_trades': grouped.size(),
            'total_profit_x': grouped['profit'].sum(),
            'win_rate': grouped.apply(lambda x: (x['profit'] > 0).mean(), include_groups=False),
            'avg_profit_rate': grouped['profit_rate'].mean(),
            'std_profit_rate': grouped['profit_rate'].std().fillna(0),
            'total_commission': grouped['commission'].sum(),
            'total_lot_size': grouped['lot_size'].sum(),
            'avg_lot_size': grouped['lot_size'].mean(),
            'avg_duration_hr': grouped['duration_hr'].mean(),
            'max_win_streak': grouped.apply(self._max_win_streak),
            'max_loss_streak': grouped.apply(self._max_loss_streak),
            'profit_factor': grouped.apply(self._profit_factor),
            'risk_reward_ratio': grouped.apply(self._risk_reward_ratio),
        })

        agg_df['inv_max_loss_streak'] = -agg_df['max_loss_streak']

        agg_df.fillna(0, inplace=True)

        logger.info("Feature aggregation complete.")

        return agg_df.reset_index()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the model feature columns from aggregated df
        """
        logger.info("Extracting model features from aggregated data...")
        return df[self.features]

    def _max_win_streak(self, trades_df):
        wins = (trades_df['profit'] > 0).astype(int).values
        return self._max_consecutive_ones(wins)

    def _max_loss_streak(self, trades_df):
        losses = (trades_df['profit'] <= 0).astype(int).values
        return self._max_consecutive_ones(losses)

    def _max_consecutive_ones(self, arr):
        max_streak = 0
        current_streak = 0
        for val in arr:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _profit_factor(self, trades_df):
        profit_sum = trades_df.loc[trades_df['profit'] > 0, 'profit'].sum()
        loss_sum = abs(trades_df.loc[trades_df['profit'] < 0, 'profit'].sum())
        if loss_sum == 0:
            return np.inf if profit_sum > 0 else 0
        return profit_sum / loss_sum

    def _risk_reward_ratio(self, trades_df):
        avg_win = trades_df.loc[trades_df['profit'] > 0, 'profit'].mean()
        avg_loss = abs(trades_df.loc[trades_df['profit'] < 0, 'profit'].mean())
        if pd.isna(avg_win) or pd.isna(avg_loss) or avg_loss == 0:
            return 0
        return avg_win / avg_loss
