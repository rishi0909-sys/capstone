import os
import json
import uuid
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Analytical modules are imported lazily inside their respective handlers
# to allow FastAPI and Uvicorn to boot in < 150ms.

load_dotenv()

app = FastAPI(title="Intelligent Pattern Discovery Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Storage
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_3_clean.csv")

# In-memory session store (Dataset ID -> session state)
sessions: Dict[str, Dict[str, Any]] = {}
_CLEAN_DF_CACHE: Optional[pd.DataFrame] = None


def _get_clean_df() -> Optional[pd.DataFrame]:
    """Lazy loader and in-memory cache for the validated clean retail dataset."""
    global _CLEAN_DF_CACHE
    if _CLEAN_DF_CACHE is not None:
        return _CLEAN_DF_CACHE
    if os.path.exists(CLEAN_DATA_PATH):
        try:
            _CLEAN_DF_CACHE = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
            return _CLEAN_DF_CACHE
        except Exception as e:
            print(f"[Warning] Error loading clean retail dataset: {e}")
    return None


_FORECASTER_CACHE: Dict[str, Any] = {}
FORECAST_MODEL_ENGINE = os.getenv("FORECAST_MODEL_ENGINE", "hgb").lower()
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "dataset_manifest.json")


def _validate_dataset_manifest() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MANIFEST_PATH):
        print(f"[Warning] Dataset manifest not found at {MANIFEST_PATH}")
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to parse dataset manifest: {e}")
        return None


def _load_precomputed_forecaster(granularity: str = "D") -> Optional[Any]:
    """Loads precomputed serialized forecaster artifact (HGB production primary, XGBoost fallback)."""
    is_daily = granularity.upper() in ["D", "DAILY"]
    freq_key = "D" if is_daily else "W"
    cache_key = f"{FORECAST_MODEL_ENGINE}:{freq_key}"
    if cache_key in _FORECASTER_CACHE:
        return _FORECASTER_CACHE[cache_key]

    if FORECAST_MODEL_ENGINE == "hgb":
        hgb_fname = "hgb_overall_daily.joblib" if is_daily else "hgb_overall_weekly.joblib"
        hgb_path = os.path.join(MODELS_DIR, "production", hgb_fname)
        if os.path.exists(hgb_path):
            try:
                from src.forecasting.hist_gradient_boosting_model import HistGradientBoostingForecaster
                fc = HistGradientBoostingForecaster()
                fc.load_model_artifact(hgb_path)
                _FORECASTER_CACHE[cache_key] = fc
                return fc
            except Exception as e:
                print(f"[Warning] Error loading production HGB forecaster {hgb_fname}, falling back to XGBoost: {e}")

    fname = "xgboost_overall_daily.json" if is_daily else "xgboost_overall_weekly.json"
    mpath = os.path.join(MODELS_DIR, fname)
    if os.path.exists(mpath):
        try:
            from src.forecasting.xgboost_model import XGBoostForecaster
            fc = XGBoostForecaster()
            fc.load_model_artifact(mpath)
            _FORECASTER_CACHE[cache_key] = fc
            return fc
        except Exception as e:
            print(f"[Warning] Error loading precomputed forecaster {fname}: {e}")
    return None


def _get_precomputed_rules(
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
    max_antecedent_size: Optional[int] = None,
    max_consequent_size: Optional[int] = None,
) -> List["RuleItem"]:
    """Loads precomputed association rules artifact with parameter filtering."""
    rules_path = os.path.join(MODELS_DIR, "association_rules.json")
    if not os.path.exists(rules_path):
        return []
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)

        filtered = []
        for r in raw_rules:
            ant = r.get("antecedents", [])
            cons = r.get("consequents", [])
            sup = float(r.get("support", 0.0))
            conf = float(r.get("confidence", 0.0))
            lift = float(r.get("lift", 0.0))

            if sup < min_support or conf < min_confidence or lift < min_lift:
                continue
            if max_antecedent_size is not None and len(ant) > max_antecedent_size:
                continue
            if max_consequent_size is not None and len(cons) > max_consequent_size:
                continue

            filtered.append(
                RuleItem(
                    antecedent=ant,
                    consequent=cons,
                    support=round(sup, 4),
                    confidence=round(conf, 4),
                    lift=round(lift, 2),
                )
            )
        return filtered
    except Exception as e:
        print(f"[Warning] Error reading precomputed association rules: {e}")
        return []


def _get_precomputed_behavior() -> Optional[Dict[str, Any]]:
    """Loads verified customer behavioral segmentation artifact."""
    path = os.path.join(MODELS_DIR, "customer_behavior.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Error reading customer behavior artifact: {e}")
    return None


def _get_precomputed_insights() -> Optional[List[Dict[str, Any]]]:
    """Loads automated statistical insights artifact."""
    path = os.path.join(MODELS_DIR, "insights.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Error reading insights artifact: {e}")
    return None


# Pydantic Request / Response Models
class DatasetResponse(BaseModel):
    dataset_id: str
    row_count: int
    columns: List[str]
    suggested_mapping: Dict[str, str]


class PreprocessingRequest(BaseModel):
    dataset_id: str
    column_mapping: Optional[Dict[str, str]] = None


class PreprocessingResponse(BaseModel):
    initial_rows: int
    final_rows: int
    duplicate_rows_removed: int
    rows_removed_missing_critical: int
    unique_transactions: int
    unique_products: int
    date_range: Optional[str]
    status: str


class MiningRequest(BaseModel):
    dataset_id: str
    min_support: float = 0.01
    min_confidence: float = 0.30
    min_lift: float = 1.10
    max_antecedent_size: Optional[int] = None
    max_consequent_size: Optional[int] = None


class RuleItem(BaseModel):
    antecedent: List[str]
    consequent: List[str]
    support: float
    confidence: float
    lift: float


class MiningResponse(BaseModel):
    rule_count: int
    rules: List[RuleItem]


class TemporalAnalyzeRequest(BaseModel):
    dataset_id: str
    target_category: Optional[str] = None
    target_product: Optional[str] = None
    granularity: Optional[str] = "month"
    metric: Optional[str] = "transaction_count"


class ForecastRequest(BaseModel):
    dataset_id: str
    target_product: str
    horizon: int = 4
    granularity: str = "W"


class ForecastResponse(BaseModel):
    historical_dates: List[str]
    historical_actuals: List[float]
    test_dates: List[str]
    test_actuals: List[float]
    test_preds: List[float]
    future_dates: List[str]
    future_preds: List[float]
    mae: float
    rmse: float
    r2: float
    mape: Optional[float]
    training_period: str
    testing_period: str
    future_lower_80: Optional[List[float]] = None
    future_upper_80: Optional[List[float]] = None
    wape: Optional[float] = None
    model_name: Optional[str] = "XGBoost Regressor (Leakage-Safe)"


class InsightRequest(BaseModel):
    dataset_id: str


class ExplainRequest(BaseModel):
    antecedent: List[str]
    consequent: List[str]
    support: float
    confidence: float
    lift: float


class ExplainResponse(BaseModel):
    explanation: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    validated: bool
    error: Optional[str] = None


# Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Intelligent Pattern Discovery Engine",
        "artifacts_available": {
            "daily_xgboost": os.path.exists(os.path.join(MODELS_DIR, "xgboost_overall_daily.json")),
            "weekly_xgboost": os.path.exists(os.path.join(MODELS_DIR, "xgboost_overall_weekly.json")),
            "association_rules": os.path.exists(os.path.join(MODELS_DIR, "association_rules.json")),
            "customer_behavior": os.path.exists(os.path.join(MODELS_DIR, "customer_behavior.json")),
            "clean_dataset": os.path.exists(CLEAN_DATA_PATH)
        }
    }


def _register_dataset(df: pd.DataFrame) -> DatasetResponse:
    dataset_id = str(uuid.uuid4())
    sessions[dataset_id] = {"raw_df": df}

    from src.preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline(df)
    pipeline._map_columns()

    return DatasetResponse(
        dataset_id=dataset_id,
        row_count=len(df),
        columns=df.columns.tolist(),
        suggested_mapping=pipeline.column_mapping,
    )


@app.post("/api/dataset/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        return _register_dataset(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")


@app.post("/api/dataset/demo", response_model=DatasetResponse)
def load_demo_dataset():
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")
    if os.path.exists(CLEAN_DATA_PATH):
        file_path = CLEAN_DATA_PATH
    elif os.path.exists(sample_path):
        file_path = sample_path
    else:
        raise HTTPException(status_code=404, detail="Demo dataset not found.")

    df = pd.read_csv(file_path)
    resp = _register_dataset(df)
    # Automatically wire clean_df for instant exploration
    if file_path == CLEAN_DATA_PATH:
        sessions[resp.dataset_id]["clean_df"] = df
    return resp


@app.post("/api/preprocessing/run", response_model=PreprocessingResponse)
def run_preprocessing(req: PreprocessingRequest):
    session = sessions.get(req.dataset_id)
    if not session:
        raise HTTPException(status_code=404, detail="Dataset session not found")

    from src.preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline(session["raw_df"], req.column_mapping)
    valid, msg = pipeline.validate_required()
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    clean_df = pipeline.clean_data()
    basket_df = pipeline.get_basket_df()

    session["clean_df"] = clean_df
    session["basket_df"] = basket_df

    summary = pipeline.get_summary()

    return PreprocessingResponse(
        initial_rows=summary.get("initial_rows", 0),
        final_rows=summary.get("final_rows", 0),
        duplicate_rows_removed=summary.get("duplicate_rows_removed", 0),
        rows_removed_missing_critical=summary.get("rows_removed_missing_critical", 0),
        unique_transactions=summary.get("unique_transactions", 0),
        unique_products=summary.get("unique_products", 0),
        date_range=summary.get("date_range"),
        status="Success",
    )


@app.post("/api/mining/run", response_model=MiningResponse)
def run_mining(req: MiningRequest):
    session = sessions.get(req.dataset_id)
    if session and "basket_df" in session:
        from src.mining.fp_growth import FPGrowthMiner
        miner = FPGrowthMiner(session["basket_df"])
        freq_items, rules = miner.mine_patterns(
            min_support=req.min_support,
            min_confidence=req.min_confidence,
            min_lift=req.min_lift,
            max_antecedent_size=req.max_antecedent_size,
            max_consequent_size=req.max_consequent_size,
        )
        if rules.empty:
            session["rules"] = pd.DataFrame()
            return MiningResponse(rule_count=0, rules=[])

        session["rules"] = rules
        rules_list = []
        for _, row in rules.iterrows():
            rules_list.append(
                RuleItem(
                    antecedent=list(row["antecedents"]),
                    consequent=list(row["consequents"]),
                    support=round(float(row["support"]), 4),
                    confidence=round(float(row["confidence"]), 4),
                    lift=round(float(row["lift"]), 2),
                )
            )
        return MiningResponse(rule_count=len(rules_list), rules=rules_list)

    # Precomputed Intelligence Explorer path
    precomputed = _get_precomputed_rules(
        min_support=req.min_support,
        min_confidence=req.min_confidence,
        min_lift=req.min_lift,
        max_antecedent_size=req.max_antecedent_size,
        max_consequent_size=req.max_consequent_size,
    )
    if precomputed:
        if req.dataset_id not in sessions:
            sessions[req.dataset_id] = {}
        sessions[req.dataset_id]["precomputed_rules"] = precomputed
        return MiningResponse(rule_count=len(precomputed), rules=precomputed)

    raise HTTPException(status_code=404, detail="No basket data or precomputed rules available.")


@app.get("/api/mining/rules", response_model=List[RuleItem])
def get_rules(dataset_id: str, limit: int = 50, offset: int = 0):
    session = sessions.get(dataset_id)
    if session and "rules" in session and not session["rules"].empty:
        rules = session["rules"]
        page = rules.iloc[offset : offset + limit]
        rules_list = []
        for _, row in page.iterrows():
            rules_list.append(
                RuleItem(
                    antecedent=list(row["antecedents"]),
                    consequent=list(row["consequents"]),
                    support=round(float(row["support"]), 4),
                    confidence=round(float(row["confidence"]), 4),
                    lift=round(float(row["lift"]), 2),
                )
            )
        return rules_list

    if session and "precomputed_rules" in session:
        return session["precomputed_rules"][offset : offset + limit]

    # Precomputed direct fallback
    all_rules = _get_precomputed_rules()
    if all_rules:
        return all_rules[offset : offset + limit]

    raise HTTPException(status_code=404, detail="Rules not found")


@app.post("/api/demographics/analyze")
def analyze_demographics(req: TemporalAnalyzeRequest):
    session = sessions.get(req.dataset_id)
    if session and "clean_df" in session:
        from src.demographics.analysis import DemographicAnalyzer
        analyzer = DemographicAnalyzer(session["clean_df"])
        pref = analyzer.analyze_category_preferences()
        summary = analyzer.get_customer_summary()
        if pref is not None:
            return {
                "status": "success",
                "summary": summary,
                "preferences": pref.to_dict(orient="records"),
            }

    # Serve precomputed behavioral segmentation
    behavior = _get_precomputed_behavior()
    if behavior:
        return {
            "status": "success",
            "summary": behavior.get("summary", {}),
            "preferences": behavior.get("preferences", []),
        }

    return {"status": "unavailable", "message": "Customer behavioral data not found."}


@app.post("/api/temporal/analyze")
def analyze_temporal(req: TemporalAnalyzeRequest):
    session = sessions.get(req.dataset_id)
    df = session.get("clean_df") if session else None
    if df is None:
        df = _get_clean_df()

    if df is None:
        raise HTTPException(status_code=400, detail="Dataset not found for temporal analysis.")

    from src.temporal.analysis import TemporalAnalyzer
    analyzer = TemporalAnalyzer(df)

    target_cat = req.target_category if req.target_category and req.target_category not in ["All", "All Categories", ""] else None
    target_prod = req.target_product if req.target_product and req.target_product not in ["All", "All Products", ""] else None
    metric = req.metric or "transaction_count"
    granularity = req.granularity or "month"

    trend = analyzer.analyze_trends(
        granularity=granularity,
        metric=metric,
        target_category=target_cat,
        target_product=target_prod
    )

    if trend is None:
        return {"status": "unavailable", "message": "Date field not found or filtered dataset is empty."}

    col_name = "year_week" if granularity == "week" else "year_month"
    if col_name in trend.columns:
        trend["period"] = trend[col_name]

    # Extract categories
    categories = ["All Categories"]
    if "category" in df.columns:
        cats = sorted([c for c in df["category"].dropna().unique().tolist() if c])
        categories.extend(cats)

    # Filter products for selected category or overall
    if target_cat and "category" in df.columns:
        prod_pool = df[df["category"] == target_cat]
    else:
        prod_pool = df

    top_products = ["All Products"] + prod_pool["product_name"].value_counts().head(25).index.tolist()

    return {
        "status": "success",
        "monthly_trend": trend.to_dict(orient="records"),
        "categories": categories,
        "products": top_products,
        "selected_category": req.target_category or "All Categories",
        "selected_product": req.target_product or "All Products",
        "metric": metric,
    }


@app.post("/api/forecast/run", response_model=ForecastResponse)
def run_forecast(req: ForecastRequest):
    """
    Serves authentic time-series forecasts.
    Uses precomputed, verified XGBoost model artifacts for overall demand,
    or runs dynamic training if a custom filtered dataset/product is requested.
    """
    # 1. Validate request parameters
    granularity = req.granularity.upper() if req.granularity else "D"
    if granularity not in ["D", "DAILY", "W", "WEEKLY", "M", "MONTHLY"]:
        granularity = "D"

    max_h = 90 if granularity in ["D", "DAILY"] else 26
    horizon = min(max(1, req.horizon), max_h)

    target_norm = str(req.target_product or "").strip().lower()
    is_overall = target_norm in ["overall", "all", ""]

    # 2. Priority: Precomputed trained artifact for overall explorer
    forecaster = _load_precomputed_forecaster(granularity)
    if forecaster is not None and is_overall:
        future = forecaster.forecast_future(horizon=horizon, freq=granularity)
        results = forecaster.results or {}
        metrics = results.get("metrics")
        if not metrics and "test_actuals" in results and "test_preds" in results:
            from src.evaluation.metrics import calculate_forecast_metrics
            metrics = calculate_forecast_metrics(results["test_actuals"], results["test_preds"])
        if not metrics:
            metrics = {}

        train_dates = results.get("train_dates", [])
        test_dates = results.get("test_dates", [])

        mae = float(metrics.get("MAE", 0.0))
        rmse = float(metrics.get("RMSE", 0.0))
        r2 = float(metrics.get("R2", 0.0))
        wape = float(metrics.get("WAPE", 0.0))
        mape_val = float(metrics.get("sMAPE", wape)) if metrics.get("sMAPE") is not None else None

        return ForecastResponse(
            historical_dates=[str(d) for d in train_dates],
            historical_actuals=[round(float(x), 2) for x in results.get("train_actuals", [])],
            test_dates=[str(d) for d in test_dates],
            test_actuals=[round(float(x), 2) for x in results.get("test_actuals", [])],
            test_preds=[round(float(x), 2) for x in results.get("test_preds", [])],
            future_dates=future["future_dates"] if future else [],
            future_preds=future["future_preds"] if future else [],
            future_lower_80=future.get("future_lower_80"),
            future_upper_80=future.get("future_upper_80"),
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 4),
            mape=round(mape_val, 2) if mape_val is not None else None,
            wape=round(wape, 2),
            training_period=f"{train_dates[0]} to {train_dates[-1]}" if train_dates else "",
            testing_period=f"{test_dates[0]} to {test_dates[-1]}" if test_dates else "",
            model_name=(
                "HistGradientBoosting Regressor (Production, Leakage-Safe)"
                if forecaster.__class__.__name__ == "HistGradientBoostingForecaster"
                else "XGBoost Regressor (Leakage-Safe)"
            ),
        )

    # 3. Dynamic fallback for specific product or custom uploaded session
    session = sessions.get(req.dataset_id)
    df = None
    if session and "clean_df" in session:
        df = session["clean_df"]
    elif req.dataset_id in ["demo", "demo_retail_data", "default"]:
        df = _get_clean_df()

    if df is not None:
        if "date" not in df.columns or "quantity" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset must have 'date' and 'quantity' columns for forecasting.",
            )

        from src.forecasting.xgboost_model import XGBoostForecaster
        from src.evaluation.metrics import calculate_forecast_metrics
        dynamic_fc = XGBoostForecaster(df, target_col="quantity", time_col="date")
        categories_set = set(df["category"].dropna().unique().tolist()) if "category" in df.columns else set()
        is_cat = req.target_product in categories_set
        cat_filter = req.target_product if is_cat else None
        prod_filter = None if (is_cat or is_overall) else req.target_product

        success, msg = dynamic_fc.train_and_evaluate(product_name=prod_filter, category=cat_filter, granularity=granularity)
        if not success:
            raise HTTPException(status_code=400, detail=msg)

        metrics = calculate_forecast_metrics(
            dynamic_fc.results["test_actuals"], dynamic_fc.results["test_preds"]
        )
        future = dynamic_fc.forecast_future(horizon=horizon, freq=granularity)

        return ForecastResponse(
            historical_dates=[str(d) for d in dynamic_fc.results["train_dates"]],
            historical_actuals=[round(float(x), 2) for x in dynamic_fc.results["train_actuals"]],
            test_dates=[str(d) for d in dynamic_fc.results["test_dates"]],
            test_actuals=[round(float(x), 2) for x in dynamic_fc.results["test_actuals"]],
            test_preds=[round(float(x), 2) for x in dynamic_fc.results["test_preds"]],
            future_dates=future["future_dates"] if future else [],
            future_preds=future["future_preds"] if future else [],
            future_lower_80=future.get("future_lower_80"),
            future_upper_80=future.get("future_upper_80"),
            mae=round(metrics.get("MAE", 0.0), 2) if metrics else 0.0,
            rmse=round(metrics.get("RMSE", 0.0), 2) if metrics else 0.0,
            r2=round(metrics.get("R2", 0.0), 4) if metrics else 0.0,
            mape=round(metrics.get("WAPE", 0.0), 2) if metrics and metrics.get("WAPE") is not None else None,
            wape=round(metrics.get("WAPE", 0.0), 2) if metrics else 0.0,
            training_period=f"{dynamic_fc.results['train_dates'][0]} to {dynamic_fc.results['train_dates'][-1]}",
            testing_period=f"{dynamic_fc.results['test_dates'][0]} to {dynamic_fc.results['test_dates'][-1]}",
            model_name="XGBoost Dynamic Product Regressor",
        )

    raise HTTPException(
        status_code=404,
        detail=f"No precomputed model or active dataset found for '{req.dataset_id}'. Please ensure models are trained or load a dataset.",
    )


@app.post("/api/insights/generate")
def generate_insights(req: InsightRequest):
    session = sessions.get(req.dataset_id)
    if session and "rules" in session and not session["rules"].empty:
        from src.insights.engine import InsightEngine
        engine = InsightEngine()
        engine.generate_association_insights(session["rules"])
        if "clean_df" in session:
            from src.demographics.analysis import DemographicAnalyzer
            demo = DemographicAnalyzer(session["clean_df"])
            pref = demo.analyze_category_preferences()
            if pref is not None:
                engine.generate_demographic_insights(pref)
        return {"insights": engine.get_insights()}

    # Precomputed insights artifact
    precomputed = _get_precomputed_insights()
    if precomputed:
        return {"insights": precomputed}

    return {"insights": []}


@app.post("/api/llm/explain", response_model=ExplainResponse)
def explain_with_llm(req: ExplainRequest):
    from src.llm.explainer import LLMExplainer
    explainer = LLMExplainer()
    if not explainer.is_available():
        return ExplainResponse(
            validated=False,
            error="LLM API Key not provided or LLM not available. Please configure in .env.",
        )

    rule_data = {
        "antecedents": req.antecedent,
        "consequents": req.consequent,
        "support": req.support,
        "confidence": req.confidence,
        "lift": req.lift,
    }

    result = explainer.explain_association_rule(rule_data)

    if "error" in result:
        return ExplainResponse(
            explanation=result.get("raw_output"),
            evidence=rule_data,
            validated=False,
            error=result["error"],
        )

    return ExplainResponse(
        explanation=result["explanation"],
        evidence=rule_data,
        validated=True,
    )


@app.get("/api/collective/overview")
def get_collective_overview():
    """Returns the persisted Multi-Dataset Collective Intelligence learning report."""
    from src.collective.engine import CollectiveIntelligenceEngine
    eng = CollectiveIntelligenceEngine(MODELS_DIR, os.path.dirname(CLEAN_DATA_PATH))
    report = eng.load_report()
    if report is None:
        raise HTTPException(status_code=404, detail="Collective learning report not found.")
    return report


@app.get("/api/collective/evolution")
def get_collective_evolution(v1: str = "v1.0", v2: str = "v2.0"):
    """Returns versioned training snapshots and a before/after knowledge comparison."""
    from src.collective.engine import CollectiveIntelligenceEngine
    eng = CollectiveIntelligenceEngine(MODELS_DIR, os.path.dirname(CLEAN_DATA_PATH))
    snapshots = eng.load_snapshots()
    s1 = next((s for s in snapshots if s["version_id"] == v1), snapshots[0] if snapshots else None)
    s2 = next((s for s in snapshots if s["version_id"] == v2), snapshots[-1] if snapshots else None)
    diff = []
    if s1 and s2:
        all_cats = set(s1.get("categories", {}).keys()) | set(s2.get("categories", {}).keys())
        for c in sorted(all_cats):
            c1 = s1.get("categories", {}).get(c, {"occurrences": 0, "coverage": 0})
            c2 = s2.get("categories", {}).get(c, {"occurrences": 0, "coverage": 0})
            diff.append({
                "category": c,
                "v1_occurrences": c1["occurrences"],
                "v2_occurrences": c2["occurrences"],
                "delta_occurrences": c2["occurrences"] - c1["occurrences"],
                "v1_coverage": c1["coverage"],
                "v2_coverage": c2["coverage"],
                "status": "Newly Discovered" if c1["occurrences"] == 0 else ("Strengthened" if c2["occurrences"] > c1["occurrences"] else "Unchanged"),
            })
    return {"snapshots": snapshots, "v1": v1, "v2": v2, "category_evolution": diff}


@app.get("/api/collective/graph")
def get_collective_graph(category: Optional[str] = None, limit: int = 25):
    """Returns nodes and weighted edges for the Cross-Dataset Relationship Graph."""
    from src.collective.engine import CollectiveIntelligenceEngine
    eng = CollectiveIntelligenceEngine(MODELS_DIR, os.path.dirname(CLEAN_DATA_PATH))
    report = eng.load_report()
    if not report:
        return {"nodes": [], "edges": []}

    nodes = {}
    edges = []
    for ds in report.get("registry", []):
        nodes[ds["dataset_id"]] = {"id": ds["dataset_id"], "label": ds["dataset_name"], "type": "dataset", "records": ds["records"]}

    cats = report.get("categories", [])
    if category:
        cats = [c for c in cats if c["canonical_category"].lower() == category.lower()]
    for c in cats[:limit]:
        cid = f"cat:{c['canonical_category']}"
        nodes[cid] = {"id": cid, "label": c["canonical_category"], "type": "category", "occurrences": c["total_occurrences"], "coverage": f"{c['dataset_coverage']}/{c['total_datasets']}"}
        for contrib in c.get("contributions", []):
            edges.append({
                "source": contrib["dataset_id"],
                "target": cid,
                "weight": contrib["occurrences"],
                "label": f"{contrib['contribution_pct']}%",
                "type": "contribution",
            })

    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/api/collective/predictive")
def get_collective_predictive():
    """Returns persisted emerging patterns, statistical anomalies, and top product associations."""
    pred_path = os.path.join(MODELS_DIR, "collective", "predictive_intelligence.json")
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            return json.load(f)
    from src.collective.predictive import build_predictive_intelligence
    return build_predictive_intelligence(MODELS_DIR, os.path.dirname(CLEAN_DATA_PATH))


@app.get("/api/collective/products")
def get_collective_products(
    search: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: str = "occurrences",
    limit: int = 25,
    offset: int = 0,
):
    """Returns paginated, searchable, filterable canonical product catalogue."""
    cat_path = os.path.join(MODELS_DIR, "collective", "product_catalogue.json")
    if not os.path.exists(cat_path):
        raise HTTPException(status_code=404, detail="Product catalogue not found.")
    with open(cat_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    if search:
        q = search.strip().lower()
        products = [p for p in products if q in p["canonical_product"].lower()]
    if category and category != "All":
        products = [p for p in products if p.get("canonical_category", "").lower() == category.lower()]

    if sort_by == "coverage":
        products = sorted(products, key=lambda x: (x["dataset_coverage"], x["total_occurrences"]), reverse=True)
    else:
        products = sorted(products, key=lambda x: (x["total_occurrences"], x["dataset_coverage"]), reverse=True)

    total = len(products)
    return {"total": total, "limit": limit, "offset": offset, "products": products[offset : offset + limit]}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for plain-English Q&A grounded in collective intelligence context."""
    await websocket.accept()
    from src.llm.explainer import LLMExplainer
    explainer = LLMExplainer()

    context = {}
    for fname, key in [
        ("collective/learning_report.json", "collective_overview"),
        ("collective/predictive_intelligence.json", "predictive_and_anomalies"),
        ("customer_behavior.json", "customer_behavior"),
    ]:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if key == "collective_overview":
                        context[key] = {
                            "overview": d.get("overview"),
                            "top_categories": d.get("categories", [])[:5],
                            "discoveries": d.get("discoveries", []),
                        }
                    elif key == "customer_behavior":
                        context[key] = d.get("summary")
                    else:
                        context[key] = {
                            "emerging_patterns": d.get("emerging_patterns", [])[:5],
                            "top_anomalies": d.get("anomalies", [])[:3],
                            "top_associations": d.get("top_associations", [])[:3],
                        }
            except Exception:
                pass

    try:
        while True:
            question = await websocket.receive_text()
            answer = explainer.chat_with_context(question, context)
            await websocket.send_text(answer)
    except WebSocketDisconnect:
        pass


_CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


@app.post("/api/voice/transcribe")
def voice_transcribe(req: Dict[str, Any]):
    """Transcribes a recorded audio clip (base64) with Gemini."""
    audio = req.get("audio_base64")
    if not audio:
        raise HTTPException(status_code=400, detail="audio_base64 is required.")
    from src.llm.gemini_agent import transcribe_audio
    try:
        text = transcribe_audio(audio, str(req.get("mime_type", "audio/webm")).split(";")[0])
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"text": text}


@app.get("/api/insights/workspace")
def get_insights_workspace():
    """Returns deduplicated, categorized business insights and executive overview."""
    from src.llm.retrieval import StructuredRetriever
    retriever = StructuredRetriever(MODELS_DIR)
    return retriever.build_workspace_insights()


@app.post("/api/insights/ask")
def ask_insights_assistant(req: Dict[str, Any]):
    """Context-aware Gemini Q&A grounded in StructuredRetriever evidence."""
    question = str(req.get("question", "")).strip()
    session_id = str(req.get("conversation_id", "default"))
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    from src.llm.retrieval import StructuredRetriever
    from src.llm.explainer import LLMExplainer

    retriever = StructuredRetriever(MODELS_DIR)
    history = _CHAT_SESSIONS.setdefault(session_id, [])[-4:]

    evidence = {
        "categories": retriever.get_category_performance()[:5],
        "top_relationships": retriever.get_product_relationships(product_query="", limit=5),
        "customer_summary": retriever.get_customer_segment_summary(),
        "emerging_and_anomalies": retriever.get_emerging_and_anomalies(),
        "recent_history": history,
    }

    explainer = LLMExplainer()
    answer = explainer.chat_with_context(question, evidence)
    history.append({"q": question, "a": answer})
    _CHAT_SESSIONS[session_id] = history[-4:]

    return {
        "conversation_id": session_id,
        "answer": answer,
        "source_references": [
            "backend/models/collective/learning_report.json",
            "backend/models/association_rules.json",
            "backend/models/customer_behavior.json",
            "backend/models/collective/predictive_intelligence.json",
        ],
        "confidence": "Evidence-Grounded" if explainer.is_available() else "Fallback",
    }
