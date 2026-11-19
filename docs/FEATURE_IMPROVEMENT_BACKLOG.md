# Feature Improvement Backlog

This document outlines potential improvements, tweaks, and optimizations for the Intelligent Pattern Discovery Engine, identified during the initial code inspection.

## 1. Feature & Architecture Improvements

### IMP-001: Persistent Session State Management
* **Feature Affected:** Core Backend Data Pipeline (`main.py`)
* **Existing Behavior:** Uploaded datasets and processed data are stored in a simple Python dictionary (`sessions`).
* **Specific Limitation:** In-memory storage is not scalable, prevents load balancing, and will result in data loss if the server restarts.
* **Proposed Change:** Migrate session state to a fast, persistent datastore like Redis or a lightweight SQL database (SQLite/PostgreSQL).
* **Why it matters:** Essential for application reliability, scalability, and stateless API design.
* **Expected Benefit:** No data loss on restarts, capable of handling concurrent users without memory bloat.
* **Implementation Complexity:** Medium
* **Priority:** P1
* **Dependencies:** None
* **Acceptance Criteria:** Users can upload a dataset, the server can be restarted, and the user can still run mining and forecasting tasks on the dataset using the returned session UUID.
* **Relevant Files:** `backend/main.py`

### IMP-002: Asynchronous ML Processing
* **Feature Affected:** FP-Growth Mining and XGBoost Forecasting endpoints
* **Existing Behavior:** Heavy machine learning tasks run synchronously on the main thread, blocking the API response until completed.
* **Specific Limitation:** Large datasets will cause API timeouts and block other users.
* **Proposed Change:** Move heavy computations to a background task queue (e.g., Celery or FastAPI `BackgroundTasks`) and implement a polling/webhook mechanism.
* **Why it matters:** Improves API responsiveness and prevents the server from hanging under heavy load.
* **Expected Benefit:** Non-blocking API requests, better user experience during long-running tasks.
* **Implementation Complexity:** High
* **Priority:** P1
* **Dependencies:** Persistent Session State (IMP-001)
* **Acceptance Criteria:** Mining and forecasting endpoints return a `task_id` immediately, and a separate endpoint allows checking the task status.

## 2. AI/ML Improvements

### IMP-003: LLM Response Caching
* **Feature Affected:** Insight Engine / LLM Explainer (`backend/src/llm/explainer.py`)
* **Existing Behavior:** Every request to explain a pattern triggers a new OpenAI API call.
* **Specific Limitation:** High risk of hitting API rate limits and incurring unnecessary costs for frequently requested or identical rules.
* **Proposed Change:** Implement a caching layer (e.g., Redis or disk cache) based on the hash of the rule data (antecedent, consequent, support, confidence).
* **Why it matters:** Cost optimization and faster response times for explanations.
* **Expected Benefit:** Reduced API costs and near-instant explanations for previously queried patterns.
* **Implementation Complexity:** Low
* **Priority:** P1
* **Dependencies:** None
* **Acceptance Criteria:** Requesting the explanation for the exact same rule twice results in only one API call, with the second response served from cache.

## 3. Testing and Observability

### IMP-004: Comprehensive Unit Test Suite
* **Feature Affected:** Entire Backend Codebase (`backend/src/`)
* **Existing Behavior:** Only basic validation tests exist (`backend/tests/test_validation.py`). Core ML logic is untested automatically.
* **Specific Limitation:** Changes to data preprocessing or ML pipelines could silently break downstream analysis.
* **Proposed Change:** Implement Pytest suites for `preprocessing`, `mining`, `forecasting`, and `demographics` modules using mocked data.
* **Why it matters:** Ensures reliability, correctness of ML algorithms, and prevents regressions.
* **Expected Benefit:** Higher code quality, safer refactoring.
* **Implementation Complexity:** Medium
* **Priority:** P0
* **Dependencies:** None
* **Acceptance Criteria:** Over 80% test coverage for all modules in `backend/src/`, passing reliably in a CI environment.
* **Relevant Files:** `backend/tests/`

## 4. UX/UI Improvements

### IMP-005: Full Data Fetching and Visualization Integration
* **Feature Affected:** Frontend Dashboard and Pages (`frontend/src/app/`)
* **Existing Behavior:** Pages are scaffolded but mostly lack integration with the backend APIs.
* **Specific Limitation:** The application is not yet usable from a graphical interface.
* **Proposed Change:** Connect all Next.js pages to their respective FastAPI endpoints using React Query or standard `fetch`. Use Recharts to visualize patterns and forecasts.
* **Why it matters:** This is the core deliverable for the end-user.
* **Expected Benefit:** A fully functional, interactive web application.
* **Implementation Complexity:** High
* **Priority:** P0
* **Dependencies:** None
* **Acceptance Criteria:** Users can upload data, trigger ML models, and view the generated charts and LLM explanations purely through the UI.
