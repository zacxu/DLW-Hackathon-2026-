# anomaly_detection import usage

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