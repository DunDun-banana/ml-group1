"""
Main pipeline
- Split train/test first to avoid data leak
- Preprocess train and test separately
- Apply feature engineering separately
- Save processed train/test
- Train model & evaluate
"""

import pandas as pd
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import model_training as mt
from pathlib import Path

RAW_CSV = r"data\raw data\Hanoi Daily 10 years.csv"
TRAIN_PROCESSED_CSV = 'data/train_processed.csv'
TEST_PROCESSED_CSV = 'data/test_processed.csv'
MODEL_PATH = 'models/random_forest_model.pkl'
TEST_RATIO = 0.2
VAL_RATIO = 0.1  # 10% validation from train

# 1. Load raw data
df = dp.load_data(RAW_CSV)

# 2. Split train/test by time (example: last 20% as test)
df = df.sort_values('datetime')
split_idx = int(len(df) * (1 - TEST_RATIO))
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# 3. Preprocess each separately
train_df = dp.preprocess(train_df)
test_df = dp.preprocess(test_df)

# 4. Split train into train/val
val_idx = int(len(train_df)*(1-VAL_RATIO))
val_df = train_df.iloc[val_idx:].copy()
train_df = train_df.iloc[:val_idx].copy()

# 5. Feature engineering
train_df = fe.create_date_features(train_df)
train_df = fe.create_lag_rolling(train_df, target='temp')
train_df = fe.encode_categorical(train_df)

val_df = fe.create_date_features(val_df)
val_df = fe.create_lag_rolling(val_df, target='temp')
val_df = fe.encode_categorical(val_df, train_columns=train_df.columns)

test_df = fe.create_date_features(test_df)
test_df = fe.create_lag_rolling(test_df, target='temp')
test_df = fe.encode_categorical(test_df, train_columns=train_df.columns)

# reset index just in case
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Optional: drop rows with NaN from lag/rolling features in train/test
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

# 6. save processed train/test 
Path('data').mkdir(exist_ok=True)
train_df.to_csv(TRAIN_PROCESSED_CSV, index=False)
test_df.to_csv(TEST_PROCESSED_CSV, index=False)
print(f"Train/test processed data saved to:\n  {TRAIN_PROCESSED_CSV}\n  {TEST_PROCESSED_CSV}")

# 7. Train RandomForest model & evaluate
Path('models').mkdir(exist_ok=True)
from src import model_training as mt

# explicit call to RandomForest version
model, val_metrics, test_metrics = mt.train_random_forest(
    train_df, val_df, test_df, target='temp', 
    model_path='models/random_forest_model.pkl'
)