"""
Smoke test — run this while the API is up locally.

    python test_api.py

Requires: requests  (pip install requests)
Set TEST_BEARER_TOKEN to match your .env BEARER_TOKEN, or it reads from env.
"""

import json
import os
import sys

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TOKEN = os.getenv("BEARER_TOKEN", "")

if not TOKEN:
    print("❌  Set BEARER_TOKEN in your environment or .env before running tests.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# ── 1. Health check ──────────────────────────────────────────────────────────
print("→ Health check...", end=" ")
r = requests.get(f"{BASE_URL}/health")
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
print("✅", r.json()["status"])

# ── 2. Auth rejection ────────────────────────────────────────────────────────
print("→ Auth rejection with bad token...", end=" ")
r = requests.post(
    f"{BASE_URL}/hackrx/run",
    headers={**HEADERS, "Authorization": "Bearer wrong_token"},
    json={"documents": "https://example.com/policy.pdf", "questions": ["test"]},
)
assert r.status_code == 401, f"Expected 401, got {r.status_code}"
print("✅")

# ── 3. Real query (uses a public sample PDF) ─────────────────────────────────
# Swap this URL for any publicly accessible policy PDF you want to test with.
TEST_DOC_URL = (
    "https://www.w3.org/WAI/WCAG21/Techniques/pdf/PDF2/table-of-contents.pdf"
)

print(f"→ Querying document: {TEST_DOC_URL}")
payload = {
    "documents": TEST_DOC_URL,
    "questions": [
        "What is this document about?",
        "Are there any coverage exclusions mentioned?",
    ],
}

r = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=payload, timeout=120)
assert r.status_code == 200, f"Expected 200, got {r.status_code}\n{r.text}"

data = r.json()
assert data["success"] is True
print(f"✅  Got {len(data['answers'])} answers in {data['metadata']['processing_time_seconds']}s")

for ans in data["answers"]:
    print(f"\n  Q: {ans['question']}")
    print(f"  Decision   : {ans['decision']}")
    print(f"  Justification: {ans['justification'][:120]}...")

print("\n🎉  All tests passed.")
