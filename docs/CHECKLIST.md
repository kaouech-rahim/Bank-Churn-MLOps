# Checklist de conformité

Légende : ✅ fait et vérifié sur le poste · 🟡 fait, vérification à terminer par le groupe
(Docker, GitHub, captures) · ❌ non fait.

## 1. Grille officielle du projet (/20)

| Composante | Critère | Preuve dans le dépôt | État |
|---|---|---|---|
| 1. Conception & DVC (3) | `src/` modulaire, typage, flake8/black | `src/`, `.flake8`, `pyproject.toml` ; flake8, black, isort, mypy sans erreur | ✅ |
| | Pipeline DVC reproductible + remote | `dvc.yaml` (split, preprocess, train), `dvc.lock`, `dvc repro` exécuté | ✅ pipeline · 🟡 remote : `.dvc/config` pointe vers un dossier local absent, rétablir S3 |
| 2. MLflow (3) | Runs : hyperparamètres, métriques, artefacts, courbes | 6 runs, 5 métriques, matrice de confusion + ROC, signature, input example | ✅ |
| | Champion dans le Model Registry | `BankChurnModel`, Staging → Production, archivage, alias `champion` | ✅ |
| 3. Serving (3) | FastAPI `/predict`, `/health`, Pydantic | `src/api/`, 26 tests, uvicorn testé, latence p95 47 ms | ✅ |
| | Dockerfile léger, non-root, multi-stage | `Dockerfile` (python:3.11-slim, 2 étapes, UID 1000, HEALTHCHECK) | ✅ construit et lancé avec Docker Desktop |
| 4. CI/CD (4) | Pytest unitaires, schéma, inférence, couverture > 75 % | 154 tests ; couverture : voir rapport §6 | ✅ |
| | Workflow CI (lint + tests) + rapport CML sur PR | `.github/workflows/ci.yml`, `scripts/cml_report.py` | ✅ PR #7 : 3 jobs verts, commentaire CML publié |
| 5. Orchestration (3) | Flow en DAG avec retries | `pipelines/training_flow.py`, `drift_flow.py` ; run via worker *Completed* | ✅ |
| | CRON / Work Pool + Blue/Green | `prefect.yaml` (churn-pool, 2 CRON, deploy réussi) ; `docker-compose.yml`, `scripts/switch_traffic.py` | ✅ Prefect · ✅ Blue/Green : 496/496 requêtes OK pendant la bascule v3 → v4 |
| 6. Observabilité (4) | KS, χ², PSI | `src/monitoring/drift.py` (existant, 24 tests) | ✅ |
| | Rapport HTML + Prometheus | Evidently HTML ; `/metrics` ; `prometheus/` + alerte | ✅ 2 cibles UP, métriques de drift exposées · 🟡 capture de l'alerte *Firing* |
| | Réentraînement automatique sur seuil | `drift_flow` : critique → réentraînement → promotion v3 (réel) | ✅ |
| Bonus | Streamlit/Gradio ou Grafana | — | ❌ |

## 2. Périmètre « trinôme » (guide §2.1)

| Exigence | État |
|---|---|
| Tronc commun (DVC, MLflow, FastAPI, Dockerfile, CI, drift + test) | ✅ |
| Prefect 3.x : work pool + CRON | ✅ |
| Rapport CML avec courbes sur PR | ✅ (PR #7) |
| Simulateur de dérive | ✅ |
| Blue/Green ou Canary | ✅ (exécuté : 0 erreur sur 496 requêtes) |
| Dashboard HTML/Evidently | ✅ |
| Exportateur Prometheus | ✅ |

## 3. Livrables (guide §6.1)

| Livrable | État |
|---|---|
| `src/`, `tests/`, `pipelines/` | ✅ |
| `dvc.yaml`, `Dockerfile`, `requirements.txt`, `.github/workflows/ci.yml` | ✅ |
| README avec badges, reproduction en 3 commandes, schéma d'architecture | ✅ |
| Tag `v1.0-release` | ❌ à créer manuellement (commandes dans la réponse finale) |
| Rapport PDF 6–8 pages | ✅ `docs/RAPPORT.pdf` · 🟡 captures à insérer |
| Démo live | 🟡 script `docs/DEMO.md`, toutes les briques testées ; répétition générale à faire |

## 4. Grille TP3 (MLflow)

| Critère | État |
|---|---|
| Serveur `sqlite:///mlflow.db` + `./mlruns`, port 5000 | ✅ commande documentée ; tracking SQLite utilisé |
| Expérience créée, params et métriques loggés | ✅ (`Bank-Churn`) |
| Matrice de confusion et ROC dans `diagnostic_plots/` | ✅ |
| Signature + input example | ✅ |
| ≥ 2 runs comparables, champion identifié | ✅ (6 runs) |
| Modèle enregistré, Staging avec description QA, Production | ✅ (nom `BankChurnModel`) |
| `predict_production.py` sur `models:/…/Production` | ✅ exécuté |
| Tests Pytest MLflow | ✅ (21 tests, SQLite temporaire) |

## 5. Grille TP4 (FastAPI / Docker)

| Critère | État |
|---|---|
| `ModelService` avec `load()`, `is_ready()` + lifespan | ✅ |
| Schéma avec `Literal` + `ge/le` | ✅ |
| `/health` 200 / 503, `/info` | ✅ |
| `/predict` avec `risk_tier`, `/predict/batch` avec moyenne | ✅ |
| `python:3.11-slim`, `appuser` UID 1000, HEALTHCHECK | ✅ |
| `docker-compose.yml` opérationnel | ✅ |
| Tests health/info, prédiction valide + 422, batch | ✅ |

## 6. Captures d'écran à produire

| # | Capture | Ce qu'elle prouve | Section du rapport |
|---|---|---|---|
| 1 | MLflow *Compare* des 6 runs | tracking et comparaison | §4 |
| 2 | MLflow *Models* : versions Production / Archived | gouvernance Registry | §4 |
| 3 | Swagger : réponse 200 et réponse 422 | API + validation Pydantic | §5 |
| 4 | `docker compose ps` (healthy) et `docker images` | conteneurs non-root, image légère | §5 |
| 5 | GitHub Actions : 3 jobs verts | CI | §6 |
| 6 | Commentaire CML sur une PR | rapport CML | §6 |
| 7 | Prefect : DAG d'un run `drift-daily` | orchestration | §7 |
| 8 | Prefect : *Deployments* avec CRON | planification | §7 |
| 9 | Dashboard Evidently | rapport HTML de drift | §8 |
| 10 | Prometheus : alerte `DataDriftCritical` *Firing* | alerte | §8 |
| 11 | Résumé de `traffic_during_switch.py` | Blue/Green sans erreur (si `errors = 0`) | §8 |
