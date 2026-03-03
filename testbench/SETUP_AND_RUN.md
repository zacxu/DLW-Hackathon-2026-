# Testbench: Setup and Run Guide

This document is intended for judges to test the project quickly and consistently.

## 1) Prerequisites

- Python `3.10+`
- Internet access (for OSM road graph download in RL planner)
- GUI-enabled environment (needed for matplotlib map clicks in route planning)

## 2) Setup

From repository root:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Start Web App

```bash
cd djangoframe
python manage.py runserver
```

Open:

- `http://127.0.0.1:8000/`

## 4) Manual UI Test Flow

### A. Model Inference

1. In dashboard, upload a sample file from `SAMPLE_VIDEOS/`
2. Click `Run Inference`
3. Verify result cards appear for `violence` and `fire`

### B. Emergency Flow

1. After inference, fill optional location/notes
2. Click `Prepare Emergency Contact`
3. Verify summary text and call link appear

### C. RL Route Planner

1. Optionally set `Number of goals`
2. Click `Select On Map + Plan Route`
3. In matplotlib window, click:
   - first click = start node
   - next clicks = goal nodes
4. Verify route JSON appears in UI and route plot window opens

## 5) API Payload Examples

Located in:

- `testbench/payloads/route_plan_payload.json`
- `testbench/payloads/emergency_standalone_payload.json`
- `testbench/payloads/contact_payload.json`

## 6) Automated Smoke Test (Local, no running server required)

From repository root:

```bash
python testbench/smoke_test_local.py
```

What it checks:

- `/api/infer/` with sample media
- `/api/emergency/`
- `/api/emergency/standalone/`
- `/api/contact/`

Note:

- RL route planner is intentionally not auto-run in this smoke test because it requires interactive map clicks.

## 7) Troubleshooting

- If inference upload fails:
  - ensure model files exist in `models/`
  - try a smaller video
  - increase upload cap using `MAX_MEDIA_UPLOAD_MB`
- If route planner does not open map:
  - run in GUI-enabled desktop session
  - avoid headless terminals

