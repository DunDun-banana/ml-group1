"# ml-group1" 
# 🌤️ Hanoi Temperature Forecasting Project

This project aims to **predict daily (and hourly) temperatures in Hanoi** using various Machine Learning models.  
It is developed as part of the **Machine Learning I course**, with the goal of building an end-to-end ML product — from data collection to model deployment with a UI demo.

---

## 🚀 Project Overview

The project follows a typical machine learning pipeline:

1. **Data Collection:**  
   Collect 10 years of Hanoi weather data from [Visual Crossing Weather API](https://www.visualcrossing.com/weather-query-builder/Hanoi/us/last15days/).

2. **Data Understanding:**  
   Explore and explain 33 weather-related features (e.g., `temperature`, `humidity`, `moonphase`, etc.), and visualize trends of Hanoi temperature over time.

3. **Data Processing:**  
   Handle missing values, normalize features, encode categorical variables, and compute correlations.

4. **Feature Engineering:**  
   Create new features that improve prediction accuracy — e.g., rolling averages, lag features, or text-based weather descriptions.

5. **Model Training & Evaluation:**  
   Train and tune models (Random Forest, XGBoost, LSTM, etc.) and evaluate with RMSE, R², and MAPE.  
   Optionally, use **Optuna** for hyperparameter tuning and **ClearML** for experiment tracking.

6. **Deployment:**  
   Build an interactive demo using **Streamlit** or **Gradio** to visualize model predictions.

7. **ONNX Conversion:**  
   Convert trained models to ONNX format for efficient deployment and inference.

---

## qProject Structure

```bash
group1/
├─ data/
│  ├─ raw data                            # Raw dataset (downloaded from Visual Crossing)
|  │  ├─ Hanoi Daily 10 years.csv        
│  │  └─ hanoi_weather_data_hourly.csv 
│  ├─ hourly_to_daily_weather.csv         # Aggregrate Hourly Data to Daily data
│  ├─ latest_3_year.csv                   # Data for retraining Model, updated daily
│  ├  
│  ├─ Today_Raw_X_input.csv               # Raw Input for Realtime Prediction
│  ├─ Today_X_input.csv                   # Processed Input for Realtime Prediction
│  └─ realtime_predictions.csv            # Prediction Results
│
├─ asset/                                 # Icon, Images, ... use for UI, Report 
│  ├─ heavy_rain.png            
│  ├─ moon.png            
│  ├─ sun.png   
│  ├─ wind.png            
│  └─ ProjectWorkflow.png
│
│
├─ logs/
│  ├─ daily_rmse.txt             # Save Realtime Prediction RMSE
│  ├─ metrics.txt                # Save Today Prediction Metrics (RMSE, MAE, R^2)
│  └─ retrain_log.pkl            # Retraining History
│
├─ models/
│  ├─ Current_model.pkl          # Current used pipeline and model
│  └─ Update_model.pkl           # New model after retraining model 
│
├─ src/
│  ├─ data_preprocessing.py                         # Load, clean data, handle missing values
│  ├─ feature_engineering_daily.py                  # Create new features for 
│  ├─ feature_engineering_hourly.py                 # Aggregate hourly feature to daily
│  ├─ hourly_adjusted_feature_engineering_daily.py 
│  ├─ feature_selection.py
│  ├─ pipeline.py
│  ├─ forecasting.py
│  ├─ model_training.py          # Train ML model, save it to /models
│  ├─ model_evaluation.py        # Evaluate model (RMSE, R², etc.)
│  ├─ monitoring.py 
│  └─ app.py                     # Streamlit / Gradio app for demo UI
│
├─ main.py                       # Main script to run full pipeline
├─ Main_Report.ipynb                       # Main script to run full pipeline
|─ FINAL-DATA_UNDERSRTANDING_FIXED_1.ipynb           # Detailed Analysis on 33 Features 
├─ Detailed_Ridge_Tuning.ipynb                       # Main script to run full pipeline
├─ Detailed_LGB_Tuning.ipynb 
├─ Detailed_Hourly_Tuning.ipynb 
├
├─ requirements.txt              # List of Python dependencies
├─ .gitignore                    # Ignore unnecessary files (venv, data/raw, etc.)
└─ README.md                     # Project documentation
