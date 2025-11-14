# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from .data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, To_Category, ConditionsEncoder, WindCategoryEncoder, SeasonClassifier, ConditionsEncoderHourly
from .new_feature_engineering_daily import * 
from .feature_selection import *


def build_full_pipeline(is_linear= False, is_category_conditions=False,is_category_season=False,is_category_wind=False,
                             encoding_method_condition='target', n_seasons=5, n_quantiles=4, drop_nan=True, drop_base=True):
    """

    is_linear = False (Use for Tree Model) do not create interaction features in feature engineering
    drop_base = True Drop all base feature
    is_category_conditions=False Do not encoding conditions to numeric
    is_category_season=False    Do not encoding season to numeric
    is_category_wind=False  Do not encoding wind to numeric
    encoding_method_condition: encoding method for conditions ['target','ordinal','quantile']
    n_seasons: number of seasons for encoding season by quantile
    n_quantile: number of category for wind_category encoding

    """
    
    pipeline = Pipeline(steps=[
        # 1. Preprocessing cơ bản
        ("preprocessing", build_preprocessing_pipeline()),
        
        # 2. Feature Engineering
        ("feature_engineering", FeatureEngineeringTransformer(
            drop_nan=drop_nan, 
            is_linear=is_linear
        )),
        
        # 3. Encoding 
        ("encoding", build_encoding_pipeline(
            is_category_conditions =is_category_conditions,
            is_category_season = is_category_season,
            is_category_wind= is_category_wind,
            encoding_method_condition=encoding_method_condition,
            n_seasons=n_seasons,
            n_quantiles=n_quantiles
        )),
        
        # 4. Drop Base Features
        ("drop_base", DropBaseFeature(drop_base=drop_base))
    ])
    
    return pipeline


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
        ("object_to_category", To_Category())
    ])
    return pipeline

 
def build_encoding_pipeline(is_category_conditions=False,is_category_season=False,is_category_wind=False, encoding_method_condition='target', n_seasons=5, n_quantiles = 4):
    """
    Encoding feature
    1. ConditionsEncoder: 
        Nếu is_category = True, để nguyên conditions
        Else, encode sang numeric theo một trong các cách ['target','ordinal','quantilte'] 

    2. SeasonClassifier : 
        Tạo nhóm season dựa trên nhiệt độ trung bình các tháng, rồi chia thành n_seasons. 
        Nếu is_category = True thì chuyển season.dtype thành category 
        Else, để nguyên kết quả sau khi chia thành n_seasons

    3. WindCategoryEncoder: 
        Nếu is_category = True thì để nguyên wind_category (biến này được tạo từ winddir, phân loại thành 8 loại gió theo domain knowledge trong bước feature engineering
        Else, thì encode sang numeric theo n_qunatiles
    """

    pipeline = Pipeline(steps=[
        ("conditions_encode", ConditionsEncoder(is_category=is_category_conditions, encoding_method=encoding_method_condition)),
        ("season_encode", SeasonClassifier(n_seasons=n_seasons, is_category=is_category_season)),
        ("wind_encode", WindCategoryEncoder(is_category=is_category_wind, n_quantiles= n_quantiles))
    ])

    return pipeline


def build_encoding_pipeline_hourly(is_category_conditions=False, is_category_season=False, is_category_wind=False, 
                                  encoding_method_condition='target', n_seasons=5, n_quantiles=4,
                                  conditions_cols=None):
    """
    Encoding feature cho dữ liệu hourly với multiple conditions columns
    
    1. ConditionsEncoderHourly: 
        - Xử lý nhiều cột conditions (cond_6h_0, cond_6h_1, cond_6h_2, cond_6h_3)
        - Nếu is_category_conditions = True, để nguyên conditions
        - Else, encode sang numeric theo một trong các cách ['target','ordinal','quantile'] 

    2. SeasonClassifier: 
        Tạo nhóm season dựa trên nhiệt độ trung bình các tháng, rồi chia thành n_seasons. 
        Nếu is_category_season = True thì chuyển season.dtype thành category 
        Else, để nguyên kết quả sau khi chia thành n_seasons

    3. WindCategoryEncoder: 
        Nếu is_category_wind = True thì để nguyên wind_category
        Else, thì encode sang numeric theo n_quantiles
    """

    pipeline = Pipeline(steps=[
        ("conditions_encode", ConditionsEncoderHourly(
            conditions_cols=conditions_cols, 
            is_category=is_category_conditions, 
            encoding_method=encoding_method_condition,
            n_quantiles=n_quantiles
        )),
        ("season_encode", SeasonClassifier(
            n_seasons=n_seasons, 
            is_category=is_category_season
        )),
        ("wind_encode", WindCategoryEncoder(
            is_category=is_category_wind, 
            n_quantiles=n_quantiles
        ))
    ])

    return pipeline

def build_preprocessing_pipeline_hourly():
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
        # ("drop_low_var", DropLowVariance(threshold=0.0)),
        # ("drop_cate", DropCategorical(unique_ratio_threshold=0.9)),
        ("object_to_category", To_Category())
    ])
    return pipeline



def build_full_pipeline_hourly(is_linear= False, is_category_conditions=False,is_category_season=False,is_category_wind=False,
                             encoding_method_condition='target', n_seasons=5, n_quantiles=4, drop_nan= False, drop_base=True):
    """

    is_linear = False (Use for Tree Model) do not create interaction features in feature engineering
    drop_base = True Drop all base feature
    is_category_conditions=False Do not encoding conditions to numeric
    is_category_season=False    Do not encoding season to numeric
    is_category_wind=False  Do not encoding wind to numeric
    encoding_method_condition: encoding method for conditions ['target','ordinal','quantile']
    n_seasons: number of seasons for encoding season by quantile
    n_quantile: number of category for wind_category encoding

    """
    
    pipeline = Pipeline(steps=[
        # 1. Preprocessing cơ bản
        ("preprocessing", build_preprocessing_pipeline_hourly()),
        
        # 2. Feature Engineering
        ("feature_engineering", FeatureEngineeringTransformer(
            drop_nan=drop_nan, 
            is_linear=is_linear
        )),
        
        # 3. Encoding 
        ("encoding", build_encoding_pipeline_hourly(
            is_category_conditions=is_category_conditions,
            is_category_season = is_category_season,
            is_category_wind= is_category_wind,
            encoding_method_condition=encoding_method_condition,
            n_seasons=n_seasons,
            n_quantiles=n_quantiles
        )),
        
        # 4. Drop Base Features
        ("drop_base", DropBaseFeature(drop_base=drop_base))
    ])
    
    return pipeline



