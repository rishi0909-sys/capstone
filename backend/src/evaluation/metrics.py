import numpy as np

def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true, dtype=np.float64) - np.array(y_pred, dtype=np.float64))))

def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.array(y_true, dtype=np.float64) - np.array(y_pred, dtype=np.float64)) ** 2))

def r2_score(y_true, y_pred):
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

def calculate_forecast_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
        
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # R2 can be negative when predictions are worse than mean
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    
    # WAPE (Weighted Absolute Percentage Error) - scale-aware & handles zeros reliably
    sum_true = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0) if sum_true > 0 else 0.0
    
    # Safely calculate MAPE, handling zero division
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = None
        
    # sMAPE (Symmetric MAPE)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    s_mask = denom > 0
    smape = float(np.mean(np.abs(y_pred[s_mask] - y_true[s_mask]) / denom[s_mask]) * 100.0) if np.any(s_mask) else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "WAPE": round(wape, 2),
        "MAPE": round(mape, 2) if mape is not None else None,
        "sMAPE": round(smape, 2)
    }
