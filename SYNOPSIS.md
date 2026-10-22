# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** Plateforme MLOps de Prédiction d'Attrition Client Bancaire (Bank Churn MLOps)
* **Taille du groupe :** 3 étudiants (Trinôme)
* **Membres de l'équipe :**
  1. **Étudiant 1 (Lead Data & DVC) :** Amir Rjeb — Email : `...`
  2. **Étudiant 2 (Lead Modeling & MLflow) :** AbdRahim Kaouech — Email : `...`
  3. **Étudiant 3 (Lead Serving & CI/CD) :** Fatma Ayadi — Email : `...`
  * *L'observabilité (drift, Prometheus, dashboard) est partagée entre les 3 membres.*
* **Lien vers le Dépôt GitHub :** `https://github.com/kaouech-rahim/Bank-Churn-MLOps.git`

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** Dans le secteur bancaire, acquérir un nouveau client coûte beaucoup plus cher que fidéliser un client existant. La banque veut identifier en amont les clients susceptibles de fermer leur compte.
* **Décision assistée ou automatisée :** Si la probabilité de départ dépasse un seuil, le client est ajouté à une liste de fidélisation (offre commerciale, appel du conseiller).
* **Objectif métier quantifié (KPI) :** Détecter au moins 70% des clients qui vont partir (rappel classe "churn" >= 0.70) avec une précision >= 0.50, pour limiter les offres inutiles.

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** Classification binaire (déséquilibrée, environ 20% de churn)
* **Variable Cible (`Target`) :** `Exited` (1 = le client a quitté la banque, 0 = il est resté)
* **Source du Dataset :** https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers *(licence à vérifier sur la page avant dépôt)*
* **Volume et dimensions :** 10 000 lignes, 14 colonnes *(nombre de lignes à confirmer avant validation, minimum exigé : 2 000)*. Les colonnes `RowNumber`, `CustomerId` et `Surname` seront supprimées (aucun pouvoir prédictif).
* **Variables prédictives clés :**
  * *Variables numériques :* CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary
  * *Variables catégorielles :* Geography, Gender (+ HasCrCard, IsActiveMember en binaire)
* **Métriques ML cibles :** F1-Score (classe churn) >= 0.60, ROC-AUC >= 0.85, Rappel >= 0.70, Latence d'inférence < 150 ms

---

## 4. Architecture MLOps Prévisionnelle

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC avec remote storage local (`dvc.yaml` : ingestion, preprocessing, train, evaluate) |
| **Tracking & Registry** | MLflow Tracking Server local (`http://127.0.0.1:5000`) & Model Registry. Comparaison LogReg vs RandomForest vs XGBoost |
| **Microservice Inférence** | API REST FastAPI (`/predict` unitaire et batch, `/health`) avec Swagger automatique |
| **Validation des données** | Schémas Pydantic stricts pour les payloads d'entrée |
| **Packaging & Isolation** | Image Docker multi-stage non-root (`python:3.11-slim`) & `docker-compose.yml` |
| **Intégration Continue (CI)** | GitHub Actions : Linting (`flake8`, `black`) + Pytest (couverture > 75%) |
| **Rapports CML** | Publication automatique des courbes (ROC, matrice de confusion) et métriques en commentaire de PR |
| **Orchestration** | Prefect 3.x (Tasks atomiques avec retries, Flow d'entraînement, Work Pool, planification CRON) |
| **Déploiement Continu** | Stratégie Zero-Downtime **Blue/Green** (deux conteneurs derrière un proxy, bascule après validation) |
| **Surveillance de Dérive** | Tests Kolmogorov-Smirnov (numérique), $\chi^2$ (catégoriel) et calcul formel du PSI, dashboard HTML Evidently |
| **Export Métriques** | Format OpenMetrics Prometheus (`/metrics`) |
| **Boucle de Rétroaction** | Déclenchement automatique du réentraînement si `drift_share >= 0.30` |

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Quelles variables vont dériver en production ?** `Age` (clientèle plus âgée), `Balance` (soldes plus élevés) et `Geography` (part plus importante de clients d'un pays à fort churn).
* **Méthode d'injection de la dérive :** Module `DriftSimulator` appliquant un décalage gaussien sur `Age` et `Balance` et une permutation des proportions de `Geography` sur le flux de production simulé.
* **Seuil d'alerte :** PSI >= 0.20 sur une variable clé ou proportion de variables en dérive >= 30%.
* **Comportement autonome du système :** Émission d'une alerte Prometheus, appel automatique du Flow Prefect de réentraînement, puis mise à jour du Model Registry et bascule Blue/Green si le nouveau modèle est meilleur.

---

## 6. Planning Prévisionnel & Validation Enseignant

* [ ] **Jalon 1 (S2) :** Dépôt initialisé, Synopsis validé, Dataset importé sous DVC
* [ ] **Jalon 2 (S4) :** Pipeline DVC + MLflow fonctionnels, 1ers tests Pytest
* [ ] **Jalon 3 (S5) :** API FastAPI opérationnelle, Dockerfile validé, Workflow CI actif
* [ ] **Jalon 4 (S6) :** Flow Prefect orchestré, Détection de drift & Prometheus
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, Rapport PDF et Soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`\
> **Commentaires :**

---

## 7. Mise à jour après réalisation (octobre 2026)

Le cadrage ci-dessus est conservé tel qu'il a été soumis. Les choix finalement
implémentés diffèrent sur les points suivants :

| Point du synopsis | Réalisation effective | Raison |
|---|---|---|
| Seuil d'alerte « PSI ≥ 0,20 sur une variable clé ou ≥ 30 % de variables en dérive » | Variable en drift si test KS/χ² significatif (p < 0,05) **et** PSI ≥ 0,10 ; alerte critique si ≥ 30 % des 8 variables dérivent | Double critère : évite les fausses alertes des tests sur de gros échantillons |
| Injection du drift par « permutation » de `Geography` | Ré-échantillonnage pondéré (Allemagne à ≈ 50 %), âge + 8 ans, soldes × 1,3 | `DriftSimulator`, paramètres dans `configs/config.yaml` |
| KPI rappel ≥ 0,70 et précision ≥ 0,50 | Rappel obtenu 0,696 au seuil 0,5, précision 0,582 (F1 0,634, ROC-AUC 0,868) | KPI non modifié ; écart sur le rappel documenté dans le rapport |
| Remote DVC local | Remote à harmoniser (voir README §7) | Le chemin versionné n'existe que sur un poste |
| Étapes DVC « ingestion, preprocessing, train, evaluate » | Trois étapes : `split`, `preprocess`, `train` (l'évaluation et la comparaison au modèle en service sont faites dans `train`) | Une étape de moins à maintenir, mêmes contrôles |
| MLflow « Tracking Server local » | Backend SQLite (`mlflow.db`) et `./mlruns` ; serveur MLflow local lancé à la demande (README §4) ; les scripts écrivent directement dans la base | Aucun serveur permanent nécessaire |
| Comportement autonome : « alerte Prometheus, appel automatique du Flow de réentraînement, mise à jour du Registry et bascule Blue/Green si meilleur » | Le flow `drift_flow` détecte le drift et lance lui-même le réentraînement ; l'alerte Prometheus `DataDriftCritical` informe sans rien déclencher ; le nouveau modèle n'est promu que s'il fait au moins aussi bien que le modèle en service (ROC-AUC, rappel non dégradé) ; la bascule Blue/Green reste manuelle (`scripts/switch_traffic.py`) | Règle de promotion corrigée par la PR #11 ; bascule gardée sous contrôle humain |
| Jalon final : tag `v1.0-release` | Le tag pointe sur `f8241d6` ; la version de référence pour la remise est `main` au commit `eb8ee7a` (PR #11) | La correction de la promotion est postérieure au tag |

Rôles : la répartition prévue au §1 est conservée telle quelle ; l'activité vérifiable dans
l'historique GitHub est résumée au §10 du rapport (`docs/RAPPORT.md`).

**Versions du modèle et règle de promotion (synthèse).** V5 (RandomForest-Tuned) reste le
champion : en Production, alias `champion`. V1, V2, V4 et V5 sont le même modèle, évalué sur les
1 000 clients de test de la référence (ROC-AUC 0,868, rappel 0,696, précision 0,582) ; les
métriques enregistrées de V3 (XGBoost, réentraînée après drift simulé) proviennent d'un autre
jeu de 400 lignes et ne sont pas directement comparables. Sur un jeu commun de 3 000 clients
jamais vus (contrôle local du 21/10/2026), V5 et V3 ont le même ROC-AUC (0,862) ; V5 a un meilleur
rappel (0,712 contre 0,455), V3 une meilleure précision et une meilleure calibration. Un candidat
n'est promu que si son ROC-AUC est ≥ celui du champion + `min_improvement` et son rappel ≥ celui
du champion − `max_recall_drop` ; avec la configuration actuelle (0,0 et 0,0), il ne doit
dégrader ni l'un ni l'autre. La précision ne participe pas à la décision. Les métriques issues de
jeux différents doivent être interprétées avec prudence.
