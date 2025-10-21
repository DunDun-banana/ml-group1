# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, CategoricalEncoder
from .feature_selection import FeatureSelectionRandomForest, FeatureSelectionExtraTrees, FeatureSelectionGradientBoosting, DropHighlyCorrelated

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

def build_RF_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. select by Random Forest feature importance
    """
    fs_pipeline_RF = Pipeline(steps = [('feature_importance', FeatureSelectionRandomForest(top_k= top_k))])
    return  fs_pipeline_RF


def build_ExTree_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. select by Extra Tree feature importance
    """
    fs_pipeline_ET = Pipeline(steps = [
                                ('feature_importance', FeatureSelectionExtraTrees(top_k = top_k ))])
    return  fs_pipeline_ET

def build_GB_featture_engineering_pipeline(top_k = 30):
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1.  1. select by Gradient Boosting feature importance
    """
    fs_pipeline_GB = Pipeline(steps = [
                                ('feature_importance', FeatureSelectionGradientBoosting(top_k = 30))])
    return  fs_pipeline_GB