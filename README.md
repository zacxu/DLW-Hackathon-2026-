# Aegis Incident Console

## Prerequisites
- Install dependencies: `pip install -r requirements.txt`
- Ensure model files exist:
  - `models/violence_MobileNet.keras`
  - `models/yolo11n.pt`

## Import and run
```python
import anomaly_detection as ad

results = ad.run_all("PATH TO THE VIDEO INPUT")
print(results)
```

`run_all(...)` returns a list of dicts:
- `detected_anomaly`
- `model_used`
- `confidence`

You can also run it from command line and itll print the json to terminal
`python anomaly_detection.py --run-all <PATH TO VIDEO>`

From what i manually trial and errored, violence needs a high threshold (>80?) and fire needs a low threshold (>30?) tbh it kinda sucks :(

## Django Frontend (run both models from UI)
1. Activate your venv and install deps:
   `pip install -r requirements.txt`
2. Start Django:
   ```
   cd djangoframe
   python manage.py runserver
   ```
3. Open:
   `http://127.0.0.1:8000/`

Available pages:
- `/` dashboard
- `/fire/` fire-focused view
- `/smoke/` smoke/violence-focused view

Inference endpoint used by the frontend:
- `POST /api/infer/` with form-data field `media` (image or video)
- Returns both model outputs from `anomaly_detection.run_all(...)`
- Upload limit is configurable with env var `MAX_MEDIA_UPLOAD_MB` (default: `1024`).

Policy route-planning endpoint used by the frontend:
- `POST /api/route-plan/` (empty JSON body is fine)
- Optional JSON field:
  - `goal_count` (positive integer). If set, map selection expects `1 start + goal_count` clicks.
- Opens matplotlib interactive selection on the host machine:
  - click `1` start point and configured number of goal points on the road graph
  - route inference runs automatically after selection
  - then a matplotlib route window opens to visualize the policy path
- Returns route summary + node path from `Policy_rl_agent` checkpoint.

Emergency-contact endpoint used by the frontend:
- `POST /api/emergency/` with JSON body containing:
  - `results` (from `/api/infer/`)
  - `location` (optional)
  - `notes` (optional)
- Returns:
  - `incident_id`
  - `summary`
  - `call_number` (defaults to `911`, override via `EMERGENCY_NUMBER` env var)

Frontend emergency flow:
- Run inference, then open the `Emergency Actions` panel.
- Fill location/notes and click `Prepare Emergency Contact`.
- Use the generated call link and incident summary.

Standalone emergency endpoint:
- `POST /api/emergency/standalone/` with JSON body containing:
  - `location` (optional)
  - `incident_type` (optional)
  - `severity` (optional)
  - `notes` (optional)
- This works without inference results.

Contact-us endpoint:
- `POST /api/contact/` with JSON body containing:
  - `name`
  - `email`
  - `topic`
  - `message`

UI additions:
- Media preview before inference.
- Result cards with confidence meters and JSON download.
- FAQ section with search.
- Contact us form with ticket IDs.
- Standalone emergency section with summary + call link.
