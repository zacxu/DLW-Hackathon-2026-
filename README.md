# anomaly_detection import usage

## Prerequisites
- Install dependencies: `pip install -r requirements.txt`
- Ensure model files exist:
  - `models/violence_MobileNet.keras`
  - `models/yolo11n.pt`

## Import and run
```python
import anomaly_detection as ad

results = ad.run_all("path/to/video.avi")
print(results)
```

`run_all(...)` returns a list of dicts:
- `detected_anomaly`
- `model_used`
- `confidence`
