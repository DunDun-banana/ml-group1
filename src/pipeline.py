# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, CategoricalEncoder
from .feature_selection import FeatureSelectionRandomForest, FeatureSelectionExtraTrees, FeatureSelectionGradientBoosting, FeatureSelectionLightGBM, DropHighlyCorrelated, SelectPercentileMutualInfoRegression

def build_preprocessing_pipeline():
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. Xử lý missing
        3. Drop low variance
        5. Drop categorical ít ý nghĩa
        6. Encode categorical
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("drop_low_var", DropLowVariance(threshold=0.0)),
        ("drop_cate", DropCategorical(unique_ratio_threshold=0.9)),
        ("condition_encode", CategoricalEncoder(columns=['conditions']))
    ])
    return pipeline


## Đối với các model không có tree feature importance thì có thể dùng cái này
def build_general_feature_selection(top_percent = 90):
    """
    Sử dụng SelectPercentileMutualInfoRegression
    Chọn top X% feature có thông tin cao nhất với target (dựa trên mutual info).
    """
    general_pipeline = Pipeline(steps = [("drop_high_correl", DropHighlyCorrelated()),
                                         ('top_percentile_feature', SelectPercentileMutualInfoRegression(percentile= top_percent))])
    return  general_pipeline


# Đối với các Tree base model thì đều có feature importance riêng, 
# dùng tree nào thì chọn importance của Tree đó nhé
def build_RF_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. select by Random Forest feature importance
    """
    pipeline2 = Pipeline(steps= [
        ("drop_high_correl", DropHighlyCorrelated()),
        ("feature_importance_select", FeatureSelectionRandomForest(top_k= top_k))
    ])
    return  pipeline2


def build_ExTree_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. select by Extra Tree feature importance
    """
    pipeline2 = Pipeline(steps= [
        ("drop_high_correl", DropHighlyCorrelated()),
        ("feature_importance_select", FeatureSelectionExtraTrees(top_k= top_k))
    ])
    return  pipeline2

def build_GB_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1.  1. select by Gradient Boosting feature importance
    """
    pipeline2 = Pipeline(steps= [
        ("drop_high_correl", DropHighlyCorrelated()),
        ("feature_importance_select", FeatureSelectionGradientBoosting(top_k= top_k))
    ])
    
    return  pipeline2

def build_LGBM_feature_engineering_pipeline(top_k = 30):
    pipeline2 = Pipeline(steps= [
        # Bước 1: Loại bỏ đặc trưng có tương quan cao
        ("drop_high_correl", DropHighlyCorrelated()),
        
        # Bước 2: Chọn lọc đặc trưng bằng LightGBM Regressor
        ("feature_importance_select", FeatureSelectionLightGBM(top_k= top_k))
    ])
    
    return pipeline2
