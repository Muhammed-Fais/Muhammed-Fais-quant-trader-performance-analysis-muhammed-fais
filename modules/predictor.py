import joblib
import numpy as np
from helpers.logger import setup_logger

logger = setup_logger("Predictor")

class Predictor:
    def __init__(self, model_path: str, scaler_path: str):
        logger.info("Loading model and scaler...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logger.info("Running predictions...")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
