import json
from modules.predictor import Predictor
from helpers.logger import setup_logger

logger = setup_logger("Main")

def main():

    predictor = Predictor(
        model_path="models/model.pkl",
        scaler_path="models/scaler.pkl"
    )

    with open("data/sample_raw_trades.json", "r") as f:
        raw_trades = json.load(f)

    preds = predictor.predict_from_raw(raw_trades)

    results = [
        {
            "user_id": user_id,
            "predicted_performance_class": "higher performer" if int(label) == 1 else "lower performer",
            "predicted_label": int(label)
        }
        for user_id, label in preds.items()
    ]

    logger.info(f"Model Prediction : {results}")

if __name__ == '__main__':
    main()