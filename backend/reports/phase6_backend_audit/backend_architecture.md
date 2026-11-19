# Backend Architecture

- **API**: `backend/main.py` (FastAPI)
- **Forecasting**: `HistGradientBoostingForecaster` (primary) + `XGBoostForecaster` (fallback), cached in `_FORECASTER_CACHE`
- **Mining**: `FPGrowthMiner`
- **Customer Analytics**: `DemographicAnalyzer`
- **Insights**: `InsightEngine`
