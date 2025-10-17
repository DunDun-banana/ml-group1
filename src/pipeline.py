# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, DropHighlyCorrelated, CategoricalEncoder


def build_preprocessing_pipeline():
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. Xử lý missing
        3. Drop low variance
        4. Drop highly correlated
        5. Drop categorical ít ý nghĩa
        6. Encode categorical
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("drop_low_var", DropLowVariance(threshold=0.0)),
        ("drop_high_corr", DropHighlyCorrelated(threshold=0.95, target_col='temp')),
        ("drop_cate", DropCategorical(unique_ratio_threshold=0.9)),
        ("condition_encode", CategoricalEncoder(columns=['conditions']))
    ])
    return pipeline
