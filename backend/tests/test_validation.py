
import pandas as pd
from src.evaluation.metrics import calculate_forecast_metrics
from src.preprocessing.pipeline import PreprocessingPipeline

def test_forecast_metrics():
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]
    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAPE"] == 0
    assert metrics["R2"] == 1.0

def test_forecast_metrics_zeros():
    y_true = [0, 0, 10]
    y_pred = [0, 0, 5]
    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert metrics["MAE"] > 0
    assert metrics["MAPE"] == 50.0 # Only calculates on non-zero true values
    
def test_preprocessing_validation():
    # Missing product_name
    df = pd.DataFrame({"transaction_id": [1, 2]})
    pipeline = PreprocessingPipeline(df)
    valid, msg = pipeline.validate_required()
    assert valid is False
    
    # Has required
    df2 = pd.DataFrame({"transaction_id": [1, 2], "product_name": ["A", "B"]})
    pipeline2 = PreprocessingPipeline(df2)
    valid, msg = pipeline2.validate_required()
    assert valid is True
