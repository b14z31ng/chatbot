import requests
import json

BASE = "http://localhost:8000"

# 1. Login
print("=== LOGIN ===")
r = requests.post(f"{BASE}/login", json={"username": "admin", "password": "admin"})
token = r.json()["access_token"]
print(f"Token: {token[:30]}...")

# 2. Upload PDF
print("\n=== UPLOAD ===")
with open("data/test_doc.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.pdf", f, "application/pdf")},
    )
print(f"Upload: {r.status_code} -> {r.json()}")

# 3. Test queries
queries = [
    "what is company name",
    "who is client",
    "what is this pdf about",
    "what technologies are used",
]
for q in queries:
    print(f'\n=== CHAT: "{q}" ===')
    r = requests.post(
        f"{BASE}/chat",
        json={"message": q},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    data = r.json()
    print(f'Answer: {data["answer"]}')
    print(f'Citations: {data["citations"]}')
    print(f'Confidence: {data["confidence"]}')
