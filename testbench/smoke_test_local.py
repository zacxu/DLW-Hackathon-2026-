from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _setup_django(repo_root: Path) -> None:
    django_project_root = repo_root / "djangoframe"
    sys.path.insert(0, str(django_project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoframe.settings")

    import django

    django.setup()


def _post_json(client, path: str, payload: dict, host: str) -> tuple[int, dict]:
    response = client.post(
        path,
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_HOST=host,
    )
    body = json.loads(response.content.decode("utf-8"))
    return response.status_code, body


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sample_video = repo_root / "SAMPLE_VIDEOS" / "factory fire.mp4"

    if not sample_video.exists():
        print(f"[FAIL] Missing sample video: {sample_video}")
        return 1

    _setup_django(repo_root=repo_root)
    from django.test import Client

    client = Client()
    host = "127.0.0.1"

    print("[1/4] Testing /api/infer/ ...")
    with sample_video.open("rb") as media_file:
        infer_response = client.post(
            "/api/infer/",
            data={"media": media_file},
            HTTP_HOST=host,
        )
    infer_body = json.loads(infer_response.content.decode("utf-8"))
    if infer_response.status_code != 200 or not infer_body.get("success"):
        print(f"[FAIL] /api/infer/ status={infer_response.status_code} body={infer_body}")
        return 1
    results = infer_body.get("results", [])
    print(f"[OK] /api/infer/ returned {len(results)} result rows")

    print("[2/4] Testing /api/emergency/ ...")
    emergency_payload = {
        "results": results,
        "location": "Judge Test Location",
        "notes": "Smoke test from testbench/smoke_test_local.py",
    }
    status, body = _post_json(client, "/api/emergency/", emergency_payload, host)
    if status != 200 or not body.get("success"):
        print(f"[FAIL] /api/emergency/ status={status} body={body}")
        return 1
    print("[OK] /api/emergency/ returned summary")

    print("[3/4] Testing /api/emergency/standalone/ ...")
    status, body = _post_json(
        client,
        "/api/emergency/standalone/",
        {
            "location": "Judge Standalone Test",
            "incident_type": "Fire/Smoke",
            "severity": "medium",
            "notes": "Standalone endpoint test",
        },
        host,
    )
    if status != 200 or not body.get("success"):
        print(f"[FAIL] /api/emergency/standalone/ status={status} body={body}")
        return 1
    print("[OK] /api/emergency/standalone/ returned summary")

    print("[4/4] Testing /api/contact/ ...")
    status, body = _post_json(
        client,
        "/api/contact/",
        {
            "name": "Judge",
            "email": "judge@example.com",
            "topic": "Integration Support",
            "message": "Automated smoke test message.",
        },
        host,
    )
    if status != 200 or not body.get("success"):
        print(f"[FAIL] /api/contact/ status={status} body={body}")
        return 1
    print("[OK] /api/contact/ accepted message")

    print("[PASS] Local smoke test completed successfully.")
    print(
        "Note: RL route planner endpoint (/api/route-plan/) is not auto-tested here "
        "because it requires interactive matplotlib clicks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
