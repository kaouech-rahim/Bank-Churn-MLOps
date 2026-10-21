"""Envoie des requetes /predict en continu pendant une bascule Blue/Green.

A lancer dans un 1er terminal, puis lancer switch_traffic.py dans un 2e terminal.
Chaque reponse est enregistree (heure, code HTTP, version du modele) dans un CSV.
Le resume final indique le nombre d'erreurs : "zero interruption" n'est
demontre que si ce nombre vaut 0 ET que la version servie a change.

Usage :
    poetry run python scripts/traffic_during_switch.py --duration 60
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import httpx

CUSTOMER = {
    "CreditScore": 619,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 58,
    "Tenure": 2,
    "Balance": 125000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 101348.88,
}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resume les reponses : total, erreurs, versions vues dans l'ordre.

    Args:
        records: Lignes {"t": ..., "status": int, "version": str}.

    Returns:
        Dict[str, Any]: total, ok, errors, status_counts, versions (ordre d'apparition).
    """
    statuses = Counter(r["status"] for r in records)
    versions: List[str] = []
    for r in records:
        if r["status"] == 200 and r["version"] not in versions:
            versions.append(r["version"])
    ok = statuses.get(200, 0)
    return {
        "total": len(records),
        "ok": ok,
        "errors": len(records) - ok,
        "status_counts": dict(statuses),
        "versions": versions,
    }


def run(url: str, duration: float, pause: float, out: Path) -> Dict[str, Any]:
    """Interroge l'API pendant `duration` secondes et ecrit le CSV."""
    records: List[Dict[str, Any]] = []
    end = time.monotonic() + duration
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < end:
            row: Dict[str, Any] = {
                "t": round(time.time(), 3),
                "status": 0,
                "version": "",
            }
            try:
                response = client.post(f"{url}/predict", json=CUSTOMER)
                row["status"] = response.status_code
                if response.status_code == 200:
                    row["version"] = response.json()["model_version"]
            except httpx.HTTPError as err:
                row["version"] = type(err).__name__  # erreur reseau = status 0
            records.append(row)
            time.sleep(pause)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "status", "version"])
        writer.writeheader()
        writer.writerows(records)
    return summarize(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trafic continu pendant la bascule")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--out", default="reports/demo_switch.csv")
    args = parser.parse_args()
    summary = run(args.url, args.duration, args.pause, Path(args.out))
    print(summary)
    sys.exit(0 if summary["errors"] == 0 else 1)
