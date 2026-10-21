# Script de démonstration — 5 minutes

Objectif : montrer la chaîne complète **API dans Docker → Prefect → injection de drift →
réentraînement → alerte Prometheus**. Chaque étape indique ce qui a été vérifié sur le
poste de développement et le plan B si une étape échoue le jour J.

| Étape | Vérifié sur le poste de dev ? |
|---|---|
| API (uvicorn), `/predict`, `/metrics` | Oui |
| Docker, Nginx Blue/Green, Prometheus | Oui : bascule v3 → v4, 496/496 requêtes OK, cibles Prometheus UP |
| Prefect : serveur, work pool, deploy, worker, flows | Oui |
| Drift `strong` + réentraînement + promotion | Oui (v3 promue, ROC-AUC holdout 0,8457 contre 0,8067) |

---

## Préparation (la veille, ~15 min)

Tous les terminaux sont ouverts à la racine du projet, avec :
```powershell
$env:PYTHONIOENCODING = "utf-8"; $env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
```

1. `poetry run dvc repro` puis `poetry run python -m src.models.train` (modèle en Production).
2. `poetry run python scripts/switch_traffic.py --init` puis `docker compose up -d --build`.
3. Vérifier : `docker compose ps` → 4 services *running*, `api_blue`/`api_green` *healthy*.
4. Terminal T1 : `poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000`
5. Terminal T2 : `poetry run prefect server start`
6. Terminal T3 : `poetry run prefect work-pool create churn-pool --type process` (une fois),
   `$env:PROJECT_DIR=(Get-Location).Path; poetry run prefect --no-prompt deploy --all`,
   puis `poetry run prefect worker start --pool churn-pool`.
7. Lancer une fois le scénario `none` pour que Prometheus affiche un état « sain » :
   `poetry run prefect deployment run "bank-churn-drift-monitoring/drift-daily" --param scenario=none`
8. Onglets navigateur prêts : Swagger `localhost:8000/docs`, MLflow `127.0.0.1:5000`,
   Prefect `127.0.0.1:4200`, Prometheus `localhost:9090/alerts`.

---

## Déroulé (5 minutes)

**0:00 – 1:00 · L'API dans Docker**
- `docker compose ps` : 4 conteneurs, l'API tourne sous `appuser` (`docker compose exec api_blue id`).
- Swagger → `POST /predict` avec l'exemple pré-rempli → `churn`, `churn_probability`,
  `risk_tier`, `model_version`.
- Changer `Geography` en `"Italy"` → **422** : Pydantic bloque la requête avant le modèle.

**1:00 – 2:00 · MLflow**
- Expérience `Bank-Churn` : sélectionner les 6 runs → *Compare* (ROC-AUC, F1).
- Onglet *Models* → `BankChurnModel` : version en Production, versions archivées.

**2:00 – 3:30 · Prefect : drift et réentraînement**
- UI Prefect → *Deployments* : `training-weekly` (CRON lundi 3 h) et `drift-daily` (6 h).
- Lancer : `poetry run prefect deployment run "bank-churn-drift-monitoring/drift-daily" --param scenario=strong`
- Montrer le DAG du run : `lot-de-production → detection-drift → rapports-drift →
  entrainement → evaluation → score-modele-actuel → enregistrement-registry → export-modele`.
- Logs attendus : `drift_share=0.375 critique=True variables=['Age', 'Balance', 'Geography']`
  puis la ligne `ROC-AUC holdout : nouveau=… / actuel=… -> promotion=…`.
- Ouvrir `reports/drift_report.html` (dashboard Evidently).

**3:30 – 4:30 · Prometheus et Blue/Green**
- `localhost:9090` → requête `churn_drift_share` puis onglet *Alerts* : `DataDriftCritical`
  passe *Pending* puis *Firing* (règle `for: 1m` : prévoir 1 à 2 minutes).
- T4 : `poetry run python scripts/traffic_during_switch.py --duration 40`
- T5 : `poetry run python scripts/switch_traffic.py` → bascule vers green avec le nouveau modèle.
- Résumé affiché par T4 : nombre d'erreurs et versions servies (ex. `['1', '3']`).
  Ne parler de « zéro interruption » **que si** `errors` vaut 0.

**4:30 – 5:00 · Conclusion**
- `poetry run python scripts/switch_traffic.py --rollback` (retour arrière en une commande).
- Rappeler : tests verts en CI, couverture, rapport CML sur PR.

---

## Plans B

| Problème | Alternative honnête |
|---|---|
| Docker ne démarre pas | `poetry run uvicorn src.api.main:app --port 8000` et démo Swagger en local ; dire que la partie conteneurs est montrée par le `Dockerfile` et le job CI `docker` |
| Alerte Prometheus pas encore *Firing* | Montrer `curl localhost:8001/metrics` → `churn_drift_critical 1.0` |
| Réentraînement trop long | Lancer le run 5 min avant la démo et montrer le run terminé dans l'UI |
| Le nouveau modèle n'est pas promu | C'est un comportement voulu : la règle protège la production ; montrer la ligne de log de comparaison |
| Pas de réseau | Tout fonctionne hors ligne sauf le `git clone` / `dvc pull` |
