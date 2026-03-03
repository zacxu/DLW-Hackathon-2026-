# Aegis Incident Console

Aegis Incident Console is a Django-based incident monitoring demo that combines:

1. **Anomaly detection inference** on uploaded media (violence + fire models)
2. **Emergency action workflows** (inference-driven + standalone)
3. **RL policy route planning** using the `Policy_rl_agent` module on a Singapore road graph


## Repository Structure

- `djangoframe/` Django project and frontend UI
- `anomaly_detection.py` combined model inference entrypoint
- `models/` local model weights (`violence_MobileNet.keras`, `fire.pt`, etc.)
- `Policy_rl_agent/` reinforcement-learning route planning module
- `SAMPLE_VIDEOS/` sample media for testing
- `testbench/` judge test assets and step-by-step test instructions

## Requirements

- Python `3.10` recommended
- Windows/macOS/Linux
- Internet access (required for OSMnx graph download used by route planning)
- GUI-capable environment for matplotlib interactive route selection
  - RL route planning requires clicking points on a map window

## Dependencies

Install all dependencies from:

```bash
pip install -r requirements.txt
```

Main libraries used:
- `django`
- `tensorflow`, `keras`
- `torch`
- `ultralytics`
- `opencv-python`
- `osmnx`, `networkx`, `gymnasium`
- `matplotlib`

## Model Files

Expected model files in `models/`:

- `models/violence_MobileNet.keras`
- `models/fire.pt`

## Quick Setup (Local)

1. Clone repo and enter folder
2. Create and activate virtual environment
3. Install dependencies
4. Run Django server

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd djangoframe
python manage.py runserver
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd djangoframe
python manage.py runserver
```

Open:

- `http://127.0.0.1:8000/`

## UI Features

- Media upload + combined inference (`/api/infer/`)
- Risk threshold slider and result cards
- Emergency summary generation from inference output (`/api/emergency/`)
- Standalone emergency summary (`/api/emergency/standalone/`)
- Contact form endpoint (`/api/contact/`)
- RL route planner (`/api/route-plan/`) with interactive matplotlib map selection

## API Endpoints

- `POST /api/infer/`
  - form-data: `media` (image/video)
  - returns combined model results
- `POST /api/route-plan/`
  - optional JSON: `goal_count` (positive integer), `show_route_plot` (bool)
  - opens interactive map for selecting `1 start + K goals`
  - runs policy inference and returns route details
- `POST /api/emergency/`
  - JSON: `results`, optional `location`, `notes`
- `POST /api/emergency/standalone/`
  - JSON: optional `location`, `incident_type`, `severity`, `notes`
- `POST /api/contact/`
  - JSON: `name`, `email`, `topic`, `message`

## Environment Variables

- `EMERGENCY_NUMBER` (default: `911`)
- `MAX_MEDIA_UPLOAD_MB` (default: `1024`)
- `POLICY_AGENT_CONFIG` (optional override config path)
- `POLICY_AGENT_CHECKPOINT` (optional override checkpoint path)

## Judge/Test Instructions

Use the dedicated folder:

- [testbench/SETUP_AND_RUN.md](./testbench/SETUP_AND_RUN.md)

It contains:
- full setup steps
- quick manual UI test flow
- API test payloads
- local smoke test command


