# Actionable Next-Step Plan

Based on the project inspection, the following action plan outlines the top five recommended improvements to elevate the Intelligent Pattern Discovery Engine from its current development stage to a robust, usable application.

## Top 5 Recommended Improvements

### 1. Connect Frontend UI to Backend Data (IMP-005)
* **Why it should be considered:** The backend APIs are fully implemented, but the frontend consists mostly of empty routes. Connecting them is critical for usability and demonstrating the project's value.
* **Dependencies and Prerequisites:** Backend server must be running.
* **Suggested Implementation Order:** #1 (Highest Priority - Core Feature)
* **Estimated Effort:** 20-30 hours (Assumes familiarity with React Query/Next.js and Recharts).
* **Acceptance Criteria:** A user can successfully upload the demo dataset from the frontend, run the FP-growth mining, and view the generated association rules and LLM explanations on the screen.

### 2. Implement Comprehensive Unit Test Suite (IMP-004)
* **Why it should be considered:** Machine learning pipelines are fragile and susceptible to data shape changes. Robust tests ensure that data preprocessing and algorithm outputs remain correct as the code evolves.
* **Dependencies and Prerequisites:** None.
* **Suggested Implementation Order:** #2 (Critical for stability).
* **Estimated Effort:** 10-15 hours.
* **Acceptance Criteria:** At least 80% coverage on `backend/src/` modules, specifically targeting `preprocessing.py`, `fp_growth.py`, and `xgboost_model.py`.

### 3. Persistent Session State Management (IMP-001)
* **Why it should be considered:** The current in-memory dictionary for storing dataframes will not scale and causes data loss on server restarts. It's a significant architectural flaw for a production app.
* **Dependencies and Prerequisites:** Selection of a datastore (Redis recommended for speed).
* **Suggested Implementation Order:** #3 (Architecture fix).
* **Estimated Effort:** 8-12 hours.
* **Acceptance Criteria:** Session data survives a `uvicorn` restart, and the application can handle multiple distinct dataset uploads simultaneously without memory leaks.

### 4. Implement LLM Response Caching (IMP-003)
* **Why it should be considered:** Calling the OpenAI API for every explanation request is slow and expensive. Since explanations for specific patterns are deterministic, they should be cached.
* **Dependencies and Prerequisites:** None (Can use basic disk cache or Redis if IMP-001 is completed first).
* **Suggested Implementation Order:** #4 (Optimization/Cost Saving).
* **Estimated Effort:** 4-6 hours.
* **Acceptance Criteria:** Requesting an explanation for an identical association rule twice results in only one API call.

### 5. Asynchronous ML Processing (IMP-002)
* **Why it should be considered:** Running heavy pandas and xgboost operations synchronously on the FastAPI main thread will block the server and cause timeouts for large datasets.
* **Dependencies and Prerequisites:** Persistent Session State Management (IMP-001).
* **Suggested Implementation Order:** #5 (Scalability).
* **Estimated Effort:** 15-20 hours.
* **Acceptance Criteria:** Long-running tasks return a background task ID immediately, and the frontend polls for completion.

## What Should Be Postponed (Scope Creep Prevention)
To ensure the core objectives are met by the deadline, the following items should be postponed to future iterations:
* **User Authentication & Authorization:** Focus on the core analytical pipeline first. Adding Auth at this stage adds unnecessary complexity.
* **Real-time Data Streaming:** The system is currently batch-based (CSV uploads). Moving to Kafka/real-time streaming is a massive architectural shift and out of scope for the current milestone.
* **Complex Custom Dashboards:** Stick to the predefined pages (Patterns, Forecasting, Demographics) before allowing users to build custom drag-and-drop dashboard views.
