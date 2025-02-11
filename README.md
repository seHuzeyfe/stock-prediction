# Stock Prediction using LSTM

## Project Overview
This repository contains code for stock price prediction using a Bidirectional LSTM model. The project includes data preprocessing, feature engineering, model development, and evaluation steps.

## Repository Structure
```
STOCK_PREDICTION/
│── data/
│   ├── processed/           # Processed datasets
│   ├── raw/                 # Raw data
│── models/                  # Saved models
│── notebooks/
│   ├── data_analyse.ipynb   # Exploratory Data Analysis
│── src/
│   ├── data/                # Data processing scripts
│   ├── features/            # Feature engineering scripts
│   ├── models/              # Model training and prediction
│   ├── preprocess/          # Data preprocessing scripts
│── stock_prediction_venv/   # Virtual environment (optional)
│── README.md
│── requirements.txt         # Dependencies
```

## Setup Instructions and File Run Order
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/STOCK_PREDICTION.git
cd STOCK_PREDICTION
```

### 2. Create a Virtual Environment
```bash
python -m venv stock_prediction_venv
source stock_prediction_venv/bin/activate   # On Mac/Linux
stock_prediction_venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
```bash
python src/data/download_data.py
```

### 5. Run Data Preprocessing
```bash
python src/preprocess/preprocess.py
```

### 6. Feature Engineering
```bash
python src/features/technical_indicators.py
```

### 7. Further Preprocess
```bash
python src/preprocess/further_preprocess.py
python src/preprocess/further_preprocess2.py
```

### After Preprocessing, data analyse code can be runned to inspect the dataset health.
- notebooks/data_analyse.ipyng

### 8. Data Split
```bash
python src/models/train_test_split.py
```

### 9. Train the Model
```bash
python src/models/LSTM_train_predict.py
```

## Dependencies
All dependencies are listed in `requirements.txt`. Major libraries used:
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Evaluation Report
- 📊 Final Model Evaluation Metrics:
- 📉 Training MAE: 0.0570
- 📊 Validation MAE: 0.0432
- 📈 Test MAE: 0.0274
- 📉 Test RMSE: 0.0353
- 📊 Test MAPE: 4.51%
- 📈 Test R² Score: 0.9749

### Results Summary
- The model achieved exceptional performance, as evidenced by the following metrics:

- Test MAE: 0.0274: On average, the model's predictions are off by 0.0274 units from the actual closing prices, indicating high accuracy.

- Test RMSE: 0.0353: Larger errors are minimal, with a root mean squared error of 0.0353, showing consistency in predictions.

- Test MAPE: 4.51%: The model's predictions are, on average, 4.51% off from the actual values, which is highly competitive for stock price forecasting.

- Test R²: 0.9749: The model explains 97.49% of the variance in the closing prices, demonstrating its ability to capture almost all underlying patterns in the data.

- Full report of the project : 'report.pdf'

