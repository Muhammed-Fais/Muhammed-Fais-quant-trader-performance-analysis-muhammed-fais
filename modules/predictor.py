import joblib
from helpers.logger import setup_logger
from modules.preprocessor import Preprocessor

logger = setup_logger("Predictor")

class Predictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize Predictor with paths to saved model and scaler.

        Args:
            model_path (str): Filepath to saved ML model (.pkl).
            scaler_path (str): Filepath to saved scaler.
        """
        logger.info("Loading model and scaler...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.preprocessor = Preprocessor()
        logger.info("Model and scaler loaded successfully.")

    def predict_from_raw(self, raw_data: list) -> dict:
        """
        Make predictions given raw JSON-like input trades data.

        Args:
            raw_data (list): List of dict trade records.

        Returns:
            dict: Mapping user_id -> predicted class (0 or 1).
        """
        logger.info("Starting prediction from raw data...")
        try:
            agg_df = self.preprocessor.preprocess_raw(raw_data)
            features_df = self.preprocessor.extract_features(agg_df)

            logger.info("Scaling features...")
            X_scaled = self.scaler.transform(features_df)

            logger.info("Making predictions...")
            preds = self.model.predict(X_scaled)

            results = dict(zip(agg_df['user_id'], preds))
            logger.info(f"Predictions done for {len(results)} users.")
            return results

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
