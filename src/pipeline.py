# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, CategoricalEncoder
from .feature_selection import FeatureSelectionGradientBoosting, FeatureSelectionGradientBoosting1, DropHighlyCorrelated1, FeatureSelectionLGB

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


def build_GB_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1.  1. select by Gradient Boosting feature importance
    """
    pipeline2 = Pipeline(steps= [
        ("feature_importance_select", FeatureSelectionGradientBoosting(top_k= top_k))
    ])
    
    return  pipeline2

def build_GB_featture_engineering_pipeline1(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1.  1. select by Gradient Boosting feature importance
    """
    pipeline2 = Pipeline(steps= [
        ("drop_high_correl", DropHighlyCorrelated1()),
        ("feature_importance_select", FeatureSelectionGradientBoosting1(top_k= top_k))
    ])

    return  pipeline2


def build_LGBM_feature_engineering_pipeline(top_k = 30):
    pipeline2 = Pipeline(steps= [
        # Bước 1: Loại bỏ đặc trưng có tương quan cao
        ("drop_high_correl", DropHighlyCorrelated1()),
        
        # Bước 2: Chọn lọc đặc trưng bằng LightGBM Regressor
        ("feature_importance_select", FeatureSelectionLGB(top_k= top_k))
    ])
    
    return pipeline2
