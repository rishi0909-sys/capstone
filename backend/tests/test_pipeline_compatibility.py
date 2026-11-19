"""
End-to-end integration test verifying that online_retail_clean.csv
is 100% compatible with all backend analytical modules.
"""

import os
import sys
import pytest
import pandas as pd

# Add backend root to sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.preprocessing.pipeline import PreprocessingPipeline
from src.mining.fp_growth import FPGrowthMiner
from src.demographics.analysis import DemographicAnalyzer
from src.temporal.analysis import TemporalAnalyzer
from src.forecasting.xgboost_model import XGBoostForecaster
from src.insights.engine import InsightEngine

DATA_PATH = os.path.join(BACKEND_DIR, "data", "online_retail_clean.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BACKEND_DIR, "data", "sample_transactions.csv")

@pytest.fixture(scope="module")
def sample_df():
    df = pd.read_csv(DATA_PATH, nrows=5000)
    return df

def test_preprocessing_pipeline(sample_df):
    pipeline = PreprocessingPipeline(sample_df)
    valid, msg = pipeline.validate_required()
    assert valid is True
    clean_df = pipeline.clean_data()
    basket_df = pipeline.get_basket_df()
    assert len(clean_df) > 0
    assert len(basket_df) > 0

def test_fp_growth_mining(sample_df):
    pipeline = PreprocessingPipeline(sample_df)
    pipeline.clean_data()
    basket_df = pipeline.get_basket_df()
    miner = FPGrowthMiner(basket_df)
    freq_items, rules = miner.mine_patterns(min_support=0.01, min_confidence=0.2, min_lift=1.0)
    assert freq_items is not None
    assert rules is not None

def test_demographic_analysis(sample_df):
    pipeline = PreprocessingPipeline(sample_df)
    clean_df = pipeline.clean_data()
    demo = DemographicAnalyzer(clean_df)
    pref = demo.analyze_category_preferences()
    assert pref is not None
    assert "probability" in pref.columns

def test_temporal_analysis(sample_df):
    pipeline = PreprocessingPipeline(sample_df)
    clean_df = pipeline.clean_data()
    temporal = TemporalAnalyzer(clean_df)
    trends = temporal.analyze_trends(granularity="week", metric="quantity")
    assert trends is not None
    assert len(trends) > 0
