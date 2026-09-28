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
* **Lien vers le Dépôt GitHub :** `https://github.com/mon-organisation/bank-churn-mlops`

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

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`  
> **Commentaires :**
