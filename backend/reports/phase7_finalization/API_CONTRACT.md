# Intelligent Pattern Discovery Engine — Frozen API Contract

- `GET /api/mining/rules?dataset_id=demo&limit=50&offset=0` → `List[Rule]`
- `POST /api/demographics/analyze` (`{"dataset_id": "demo"}`) → Behavioral & RFM summary + category preferences
- `POST /api/temporal/analyze` (`{"dataset_id": "demo"}`) → Temporal sales trends
- `POST /api/forecast/run` (`{"dataset_id": "demo", "granularity": "D|W", "horizon": 14, "target_product": "Overall"}`) → `ForecastResponse`
- `POST /api/insights/generate` (`{"dataset_id": "demo"}`) → Automated business insights
