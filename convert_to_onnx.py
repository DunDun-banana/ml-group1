# ==============================================================================
# SCRIPT TO CONVERT MODEL FROM PKL TO ONNX
# ==============================================================================
#
# PURPOSE:
# This script loads the file 'Complete_Pipeline_LGBM_models.pkl' which contains 5
# pairs of (preprocessing pipeline + LightGBM model). Then it separates each model
# and converts them into 5 individual .onnx files.
#
# APPROACH:
# Since `skl2onnx` cannot convert custom pipeline classes (custom transformers),
# we will apply the "baking" method:
# 1. Pass sample data through the preprocessing pipeline to obtain the final input
#    format that the LGBM model actually receives.
# 2. Convert only the LightGBM model using that "baked" input format.
#
# REQUIREMENTS BEFORE RUNNING:
# 1. Install required libraries:
#    pip install joblib pandas scikit-learn lightgbm skl2onnx onnxmltools onnxruntime
# 2. Ensure this script is placed in the project root directory.
# 3. Ensure the paths to the .pkl file and .csv file are correct.
#
# OUTPUT:
# 5 .onnx files will be saved in the 'models/' directory.
#
# ==============================================================================

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from onnxmltools import convert_lightgbm 
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
import onnxruntime as rt
import traceback

# --- STEP 0: REDEFINE CLASS SO JOBLIB CAN LOAD THE FILE ---
# This step is required so that joblib can correctly recognize and load
# the object structure stored in the .pkl file.
class CustomMultiOutputRegressor:
    def __init__(self, models_per_target, complete_pipelines_per_target):
        self.models_per_target = models_per_target
        self.complete_pipelines_per_target = complete_pipelines_per_target
        self.target_names = list(models_per_target.keys())

    def predict(self, X):
        predictions = {}
        for target_name, model_info in self.models_per_target.items():
            pipeline = self.complete_pipelines_per_target[target_name]
            X_processed = pipeline.transform(X)
            X_final = X_processed.iloc[30:]
            pred = model_info['model'].predict(X_final)
            predictions[target_name] = pred
        return pd.DataFrame(predictions)

    def get_params(self, deep=True):
        return {
            "models_per_target": self.models_per_target,
            "complete_pipelines_per_target": self.complete_pipelines_per_target
        }
    
PKL_MODELS_PATH = r"models/Complete_Pipeline_LGBM_models.pkl"
RAW_DATA_PATH = r"data/raw data/Hanoi Daily 10 years.csv"
ONNX_OUTPUT_DIR = r"models/"

# --- STEP 1: LOAD OBJECTS FROM PKL FILE ---
try:
    print(f"Loading data from: {PKL_MODELS_PATH}")
    saved_data = joblib.load(PKL_MODELS_PATH)
    print("Data loaded successfully!")

    models_per_target = saved_data['models']
    pipelines_per_target = saved_data['pipelines']
    target_names = list(models_per_target.keys())
except FileNotFoundError:
    print(f"ERROR: File '{PKL_MODELS_PATH}' not found. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR while loading pkl file: {e}")
    exit()


# --- STEP 2: PREPARE SAMPLE DATA TO INFER INPUT SHAPE ---
try:
    print("Preparing sample data...")
    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=['datetime'])
    # Basic preprocessing similar to the notebook
    df_processed = df_raw.drop(columns=['description', 'icon', 'stations', 'name'], errors='ignore')
    if 'datetime' in df_processed.columns:
        df_processed.set_index('datetime', inplace=True)
    else:
        print("ERROR: Column 'datetime' not found to set as index.")
        exit()
    X_sample_raw = df_processed.iloc[30:]  # Use the first 30 rows to support rolling calculations
    print("Sample data is ready.")
except FileNotFoundError:
    print(f"ERROR: Raw data file '{RAW_DATA_PATH}' not found.")
    exit()

# --- STEP 3: "BAKE" THE DATA THROUGH PIPELINE TO DEFINE ONNX INPUT SHAPE ---
print("Processing sample data through pipeline to determine ONNX input shape...")
sample_pipeline = pipelines_per_target[target_names[0]]
X_processed_sample = sample_pipeline.transform(X_sample_raw)

# Important: Remove the first 30 rows containing NaNs from rolling features
X_processed_sample_final = X_processed_sample.iloc[30:]
if X_processed_sample_final.empty:
    print("ERROR: Processed sample data is empty after trimming. Increase size of X_sample_raw.")
    exit()

num_features = X_processed_sample_final.shape[1]
print(f"==> Number of input features for ONNX model: {num_features}")

initial_types_for_lgbm = [('float_input', FloatTensorType([None, num_features]))]

# --- STEP 4: LOOP TO CONVERT 5 MODELS ---
conversion_success = True
for target_name in target_names:
    print(f"\n--- Converting model for target: {target_name} ---")

    lgbm_model = models_per_target[target_name]['model']

    try:
        onnx_model = convert_lightgbm(
            lgbm_model,
            'lgbm_model',
            initial_types=initial_types_for_lgbm,
            target_opset=12
        )

        onnx_filename = f"{ONNX_OUTPUT_DIR}lgbm_model_{target_name}.onnx"
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ Conversion successful! Saved to: {onnx_filename}")

    except Exception as e:
        print(f"❌ Conversion failed for {target_name}: {e}")
        traceback.print_exc()
        conversion_success = False

    print("-" * 60)

# --- STEP 5: VERIFICATION ---
if conversion_success:
    print("\n--- Starting ONNX model verification ---")
    try:
        target_to_test = target_names[0]
        onnx_path_to_test = f"{ONNX_OUTPUT_DIR}lgbm_model_{target_to_test}.onnx"
        
        print(f"Loading ONNX session from: {onnx_path_to_test}")
        sess = rt.InferenceSession(onnx_path_to_test)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        test_input_data = X_processed_sample_final.iloc[:1].to_numpy().astype(np.float32)
        
        print(f"Predicting with input shape: {test_input_data.shape}")

        result = sess.run([output_name], {input_name: test_input_data})
        
        print(f"✅ Verification successful! Prediction result: {result[0]}")
        print("\nPROCESS COMPLETED!")
        print("You can now use the .onnx files together with the preprocessing pipeline in your application.")

    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
else:
    print("\nConversion encountered errors. Please check logs above.")
