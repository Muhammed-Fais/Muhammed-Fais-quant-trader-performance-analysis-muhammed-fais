# Quant Trader Performance Analysis

A machine learning-based system for analyzing trading data and predicting trader performance. This project processes raw trading data, extracts meaningful features, and uses machine learning to classify traders as high or low performers.

## Features

- Raw trading data processing and feature engineering
- Machine learning model for trader performance classification
- Comprehensive logging system
- Data visualization and analysis capabilities
- Performance metrics calculation

## Project Structure

```
├── data/               # Raw and processed trading data
│   ├── raw/           # Original trading data
│   ├── processed/     # Processed and cleaned data
│   └── sample_raw_trades.json  # Sample trading data for testing
├── helpers/           # Utility functions and logging setup
├── logs/             # Application logs
├── models/           # Trained ML models and scalers
├── modules/          # Core functionality modules
│   ├── preprocessor.py  # Data preprocessing and feature engineering
│   └── predictor.py     # Model prediction functionality
├── notebooks/        # Jupyter notebooks for analysis
│   ├── data_analysis.ipynb                    # Initial data exploration and analysis
│   ├── trader_score_composition_metrics.ipynb # Trader Quality Score (TQS) calculation
│   ├── trader_classification_model.ipynb      # ML model for trader classification
│   └── trader_segmentation.ipynb             # Trader behavior segmentation analysis
├── reports/          # Generated reports and visualizations
│   ├── figures/      # Generated visualizations and plots
│   └── Trading-Data-Analysis-and-Performance-Insights.pdf  # Comprehensive analysis report for stakeholders
├── main.py           # Main application entry point
└── requirements.txt  # Project dependencies
```

## Data and Reports

### Sample Data
The project includes a sample dataset (`data/sample_raw_trades.json`) that demonstrates the expected format of trading data. Each trade record contains:
- `user_id`: Unique identifier for the trader
- `profit`: Trade profit/loss amount
- `profit_rate`: Profit as a percentage
- `commission`: Trading commission
- `lot_size`: Size of the trade
- `duration_hr`: Trade duration in hours

This sample data can be used to test the system and understand the data structure.

### Analysis Report
A comprehensive analysis report (`reports/Trading-Data-Analysis-and-Performance-Insights.pdf`) is included, which contains:
- Detailed analysis of trading patterns
- Performance metrics and insights
- Visualization of key findings
- Recommendations for trader evaluation
- Model performance analysis
- Segmentation results

## Notebooks Overview

The project includes several Jupyter notebooks that provide detailed analysis and implementation:

1. **data_analysis.ipynb**
   - Initial exploration of the trading dataset
   - Basic statistical analysis and data quality checks
   - Data visualization and pattern identification

2. **trader_score_composition_metrics.ipynb**
   - Implements the Trader Quality Score (TQS) calculation
   - Computes individual trader-level quantitative metrics
   - Generates composite scores for trader performance ranking
   - Includes advanced metrics like profit factor, win/loss streaks, and risk-reward ratios

3. **trader_classification_model.ipynb**
   - Implements machine learning models for trader classification
   - Uses Random Forest and XGBoost classifiers
   - Includes model evaluation metrics and visualizations
   - Features SHAP analysis for model interpretability
   - Classifies traders as high or low performers based on TQS

4. **trader_segmentation.ipynb**
   - Performs trader segmentation using K-Means clustering
   - Analyzes trader behavior patterns
   - Identifies distinct trader groups based on:
     - Risk profiles
     - Lot sizes
     - Currency pair preferences
     - Trading consistency
   - Includes visualization of segments using PCA and radar plots

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your trading data in JSON format and place it in the `data/` directory. You can use the `sample_raw_trades.json` as a template for the expected data format.

2. Run the main script:
```bash
python main.py
```

The script will:
- Load the trained model and scaler
- Process the input trading data
- Generate performance predictions
- Output results to the console and logs

## Dependencies

- pandas==2.3.0
- numpy==2.1.0
- matplotlib==3.10.3
- seaborn==0.13.2
- scikit-learn==1.7.0
- shap
- xgboost

## Development

### Project Components

- **Preprocessor**: Handles data cleaning, feature engineering, and data transformation
- **Predictor**: Manages model loading and prediction generation
- **Logger**: Provides comprehensive logging functionality
