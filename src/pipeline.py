# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, DropHighlyCorrelated,StationMultiHotEncoder, StationPreprocessor, CategoricalEncoder, DropColumns

# pipeline 1: chọn mainsation rồi label encoding, khả năng bị drop luôn ở bước low_variance
def build_preprocessing_pipeline():
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: HandleMissing → DropLowVariance → DropHighlyCorrelated 
         → DropCategorical (drop categorical chỉ có 1 unique value, hoặc gần như unique với mọi sample)
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("station", StationPreprocessor()),  # tạo main_station và encode
        ("low_var", DropLowVariance(threshold=0.0)), # thay bằng Variancethreshold của sklearn luôn cg dc
        ("corr", DropHighlyCorrelated(threshold=0.95, target_col='temp')),
        ("drop_cat", DropCategorical(unique_ratio_threshold=0.9)),
        ("cat_encode", CategoricalEncoder(columns=['icon', 'conditions'])),  # (dùng tạm ) encode categorical còn lại icon, conditions
        #("drop_stations", DropColumns(columns=['stations']))
    ])
    return pipeline

# Pipeline 2: dùng station oneHotEncoding
def build_preprocessing_pipeline_2():
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("low_var", DropLowVariance(threshold=0.0)),
        ("corr", DropHighlyCorrelated(threshold=0.95, target_col='temp')),
        ("drop_cat", DropCategorical(unique_ratio_threshold=0.9)),
        ("station", StationMultiHotEncoder(column='stations')),  # thêm bước xử lý stations
        ("cat_encode", CategoricalEncoder(columns=['icon', 'conditions']))
    ])
    return pipeline
