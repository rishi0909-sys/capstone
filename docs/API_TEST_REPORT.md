# API Test Report

## 1. Test Environment
- **Frontend URL:** `http://localhost:3000`
- **Backend URL:** `http://localhost:8000`
- **Date Tested:** 29 September 2026
- **Test Context:** Local verification of Phase 3, Phase 4, and Phase 5 implementation against FastAPI backend.

## 2. Frontend Integration Tests
| Test                   | Result            | Notes |
| ---------------------- | ----------------- | ----- |
| Dataset upload         | PASS              | `FormData` correctly posts to `/api/dataset/upload` and retrieves `dataset_id`. |
| Dataset persistence    | PASS              | Zustand `persist` middleware successfully keeps `dataset_id` in `localStorage` across refreshes. |
| Preprocessing          | PASS              | Loading state works. Metrics rendered correctly upon success. |
| Mining                 | PASS              | Association rules successfully requested via `/api/mining/run`. |
| Pattern table          | PASS              | Paginated, sortable rendering of `RuleItem`s works without fabricating scores. |
| Pattern detail         | PASS              | Statistical evidence clearly separated from LLM generation area. |
| LLM explanation        | BLOCKED           | Blocked due to external OpenAI API dependency (no valid API key in `.env`). Failure gracefully caught by `ErrorState`. |
| Forecasting run        | PASS              | Forecast successfully requested via `/api/forecast/run`. `ForecastResponse` parsed correctly. |
| Forecasting charts     | PASS              | Recharts maps historical, test, and future predictions seamlessly onto a continuous timeline. |
| Forecasting metrics    | PASS              | Real metrics (MAE, RMSE, R², MAPE) dynamically displayed without inventing data. |
| Stale session recovery | PASS              | Frontend wrapper `DatasetRequiredState` intercepts invalid state gracefully across all analytical pages. |
| Backend unavailable    | PASS              | `lib/api/client.ts` catches `fetch` failures and converts them to standardized `APIError`, handled by React Query `ErrorState`. |

## 3. Postman API Tests
| Endpoint | Method | Test | Result | Notes |
| -------- | ------ | ---- | ------ | ----- |
| `/api/dataset/upload` | POST | Positive | PASS | Uploads CSV and returns `dataset_id`. |
| `/api/preprocessing/run` | POST | Positive | PASS | Returns initial/final rows and status. |
| `/api/mining/run` | POST | Positive | PASS | Returns array of `RuleItem`. |
| `/api/llm/explain` | POST | Negative | PASS (Caught) | Properly returns API error when OpenAI key is missing. |
| `/api/forecast/run` | POST | Positive | PASS | Returns `ForecastResponse` arrays (historical/future) and metrics. |
| `/api/forecast/run` | POST | Negative | PASS (Caught) | Throws 400 when missing `dataset_id` or invalid `target_product`. |

## 4. Bugs Found
- **Bug ID 1**
  - **Feature:** LLM Explanation
  - **Expected behavior:** Explain a pattern.
  - **Actual behavior:** Fails with 500 error if API key is missing.
  - **Root cause:** External dependency.
  - **Fix:** N/A (Requires environment config).
  - **Verification status:** Graceful frontend failure verified.

## 5. Known Limitations
- **In-memory backend sessions:** Backend restarts cause dataset IDs to invalidate. Frontend gracefully prompts user to re-upload.
- **Synchronous ML operations:** Pattern mining and XGBoost forecasting block the HTTP thread until complete.
- **LLM Configuration:** Blocked locally without an OpenAI API key.
- **Demographics / Dashboard:** Deferred per Phase 5 scope cutoff.
