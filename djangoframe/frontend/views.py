import os
import sys
import tempfile
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
VIOLENCE_MODEL_PATH = MODELS_DIR / "violence_MobileNet.keras"
FIRE_MODEL_PATH = MODELS_DIR / "fire.pt"
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "911")
SUPPORTED_SUFFIXES = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".m4v",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def _load_inference_module():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import anomaly_detection

    return anomaly_detection


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        return Path(tmp.name)


def _base_context() -> dict:
    return {"emergency_number": EMERGENCY_NUMBER}


def home(request):
    return render(request, "1/base.html", _base_context())


def fire(request):
    return render(request, "2/fire.html", _base_context())


def smoke(request):
    return render(request, "2/smoke.html", _base_context())


def _parse_confidence(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_emergency_summary(incident_id: str, location: str, notes: str, results: list[dict]) -> str:
    highest = max(results, key=lambda row: _parse_confidence(row.get("confidence", 0.0)))
    top_anomaly = str(highest.get("detected_anomaly", "unknown"))
    top_confidence = _parse_confidence(highest.get("confidence", 0.0))
    lines = [
        f"Incident ID: {incident_id}",
        f"Location: {location}",
        f"Top detected anomaly: {top_anomaly}",
        f"Top confidence: {top_confidence:.6f}",
        "All model outputs:",
    ]
    for result in results:
        lines.append(
            f"- {result.get('detected_anomaly', 'unknown')}: "
            f"{_parse_confidence(result.get('confidence', 0.0)):.6f} "
            f"(model: {result.get('model_used', 'unknown')})"
        )
    if notes:
        lines.append(f"Notes: {notes}")
    return "\n".join(lines)


def _format_manual_emergency_summary(
    incident_id: str,
    location: str,
    incident_type: str,
    severity: str,
    notes: str,
) -> str:
    lines = [
        f"Incident ID: {incident_id}",
        f"Location: {location}",
        f"Incident type: {incident_type}",
        f"Severity: {severity}",
        "Trigger source: manual emergency action (standalone).",
    ]
    if notes:
        lines.append(f"Notes: {notes}")
    return "\n".join(lines)


def _new_incident_payload(incident_id: str, created_at: str, summary: str, message: str) -> dict:
    return {
        "success": True,
        "incident_id": incident_id,
        "created_at": created_at,
        "call_number": EMERGENCY_NUMBER,
        "summary": summary,
        "message": message,
    }


@require_POST
def run_inference(request):
    uploaded_file = request.FILES.get("media")
    if uploaded_file is None:
        return JsonResponse({"success": False, "error": "No file uploaded under field 'media'."}, status=400)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return JsonResponse({"success": False, "error": f"Unsupported file type: {suffix}"}, status=400)

    if not VIOLENCE_MODEL_PATH.exists() or not FIRE_MODEL_PATH.exists():
        return JsonResponse(
            {
                "success": False,
                "error": (
                    "Model file is missing. Expected "
                    f"'{VIOLENCE_MODEL_PATH}' and '{FIRE_MODEL_PATH}'."
                ),
            },
            status=500,
        )

    temp_input_path = _save_uploaded_file(uploaded_file)

    try:
        anomaly_detection = _load_inference_module()
        results = anomaly_detection.run_all(
            input_path=str(temp_input_path),
            violence_model_path=str(VIOLENCE_MODEL_PATH),
            yolo_model_path=str(FIRE_MODEL_PATH),
        )
        return JsonResponse({"success": True, "results": results})
    except Exception as exc:
        return JsonResponse({"success": False, "error": str(exc)}, status=500)
    finally:
        if temp_input_path.exists():
            os.unlink(temp_input_path)


@require_POST
def contact_emergency(request):
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON body."}, status=400)

    results = data.get("results")
    location = str(data.get("location", "")).strip() or "Unknown location"
    notes = str(data.get("notes", "")).strip()

    if not isinstance(results, list) or not results:
        return JsonResponse(
            {"success": False, "error": "Emergency contact requires non-empty inference results."},
            status=400,
        )

    incident_id = uuid.uuid4().hex[:10].upper()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = _format_emergency_summary(
        incident_id=incident_id,
        location=location,
        notes=notes,
        results=results,
    )

    request.session["last_incident"] = {
        "incident_id": incident_id,
        "created_at": created_at,
        "location": location,
        "notes": notes,
        "results": results,
    }

    return JsonResponse(
        _new_incident_payload(
            incident_id=incident_id,
            created_at=created_at,
            summary=summary,
            message="Emergency summary generated from inference results.",
        )
    )


@require_POST
def contact_emergency_standalone(request):
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON body."}, status=400)

    location = str(data.get("location", "")).strip() or "Unknown location"
    notes = str(data.get("notes", "")).strip()
    incident_type = str(data.get("incident_type", "")).strip() or "Unknown"
    severity = str(data.get("severity", "")).strip() or "medium"

    incident_id = uuid.uuid4().hex[:10].upper()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = _format_manual_emergency_summary(
        incident_id=incident_id,
        location=location,
        incident_type=incident_type,
        severity=severity,
        notes=notes,
    )

    request.session["last_manual_incident"] = {
        "incident_id": incident_id,
        "created_at": created_at,
        "location": location,
        "incident_type": incident_type,
        "severity": severity,
        "notes": notes,
    }

    return JsonResponse(
        _new_incident_payload(
            incident_id=incident_id,
            created_at=created_at,
            summary=summary,
            message="Standalone emergency contact prepared.",
        )
    )


@require_POST
def submit_contact_message(request):
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON body."}, status=400)

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip()
    topic = str(data.get("topic", "")).strip() or "General"
    message = str(data.get("message", "")).strip()

    if not name or not email or not message:
        return JsonResponse(
            {"success": False, "error": "Name, email, and message are required."},
            status=400,
        )

    ticket_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    contact_messages = request.session.get("contact_messages", [])
    contact_messages.append(
        {
            "ticket_id": ticket_id,
            "created_at": created_at,
            "name": name,
            "email": email,
            "topic": topic,
            "message": message,
        }
    )
    request.session["contact_messages"] = contact_messages[-20:]

    return JsonResponse(
        {
            "success": True,
            "ticket_id": ticket_id,
            "created_at": created_at,
            "message": "Message received. Our team will follow up shortly.",
        }
    )
