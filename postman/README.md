# Postman API Testing

This directory contains a complete Postman collection to test the FastAPI backend of the Intelligent Pattern Discovery Engine. It verifies the complete analytical pipeline from data loading to AI explanation.

## Contents
- `Intelligent-Pattern-Discovery-Engine.postman_collection.json`: The main Postman collection containing all API requests organized by workflow stages and negative tests.
- `Intelligent-Pattern-Discovery-Engine.postman_environment.json`: The environment variables needed to run the collection (`baseUrl`, `datasetId`, `OPENAI_API_KEY`).

## Instructions

1. **Start the FastAPI Backend**
   Ensure you are in the project root.
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn main:app --reload --port 8000
   ```

2. **Import into Postman**
   - Open Postman Desktop App.
   - Click "Import" and select both `.json` files from this directory.
   - Select the "Intelligent Pattern Discovery Engine Environment" from the environment dropdown in the top right.

3. **LLM Explainer Setup (Optional)**
   - If you have an OpenAI API key and want to test the LLM Explanation endpoint, edit the Environment in Postman and paste your key into the `OPENAI_API_KEY` current value. If you don't provide a key, the tests verify that the system handles it gracefully.

4. **Running the Workflow**
   - The collection uses variable cascading. When you run the `02 - Dataset / Load Demo Dataset` request, a test script automatically captures the returned `dataset_id` and saves it to the `{{datasetId}}` environment variable.
   - All subsequent requests (Preprocessing, Pattern Mining, Forecasting, etc.) automatically use this `{{datasetId}}` variable.
   - You can run the entire collection at once using Postman's "Run Collection" feature or execute them sequentially.

## Automated Verification via Newman

You can also run the complete end-to-end test suite automatically from the command line using Newman.

```bash
npm install -g newman
newman run postman/Intelligent-Pattern-Discovery-Engine.postman_collection.json -e postman/Intelligent-Pattern-Discovery-Engine.postman_environment.json
```

## Analytical Outputs Validated
The tests explicitly assert that:
- FP-Growth successfully calculates and returns association rules with valid ranges (e.g., Support & Confidence within [0,1], non-negative Lift).
- XGBoost forecasting produces numerical historical, test, and future predictions along with evaluation metrics (MAE, RMSE, R²).
- AI Insights are successfully generated based on the analytical results.
