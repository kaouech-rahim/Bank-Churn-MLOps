# Partie d'Amir : Data, DVC et détection de drift

**En une phrase :** à partir de `churn.csv`, cette partie prépare les données
(5 000 lignes pour entraîner, 5 000 gardées pour simuler la production), les
versionne avec DVC, et fournit le moteur qui détecte le data drift, un simulateur
de drift et un dashboard HTML.

---

## 1. Ce qui est fait

| Élément | Fichier | Rôle |
|---|---|---|
| Configuration | `configs/config.yaml` | Chemins, colonnes, seuils : rien n'est écrit en dur dans le code |
| Ingestion | `src/data/load_data.py` | Charge un CSV et refuse un fichier absent ou vide |
| Découpage | `src/data/split_data.py` | `churn.csv` → `churn_reference.csv` (5 000) + `churn_production_pool.csv` (5 000), même taux de churn |
| Prétraitement | `src/features/preprocess.py` | Nettoyage, validation du schéma, split train/test, encodage → `train.csv`, `test.csv`, `scaler_params.json` |
| Pipeline DVC | `dvc.yaml` | 2 étapes enchaînées : `split` puis `preprocess` |
| Détection de drift | `src/monitoring/drift.py` | Tests KS, χ² et PSI, `drift_share`, alerte `critical` |
| Simulateur | `src/monitoring/drift_simulator.py` | Fabrique un drift : âge +8 ans, soldes ×1,3, Allemagne à 50 % |
| Dashboard | `src/monitoring/dashboard.py` | Rapport Evidently `reports/drift_report.html` + résumé `.json` |
| Tests | `tests/` | 79 tests, 95 % de couverture |
| Qualité | `.pre-commit-config.yaml` | black, isort, flake8 avant chaque commit |

**Règle de drift :** une variable est « en drift » si le test (KS ou χ²) est
significatif (p < 0,05) **et** si son PSI ≥ 0,10. Alerte critique si au moins
30 % des 8 variables surveillées dérivent (soit 3 variables ou plus).

---

## 2. Démarrer (avec l'environnement virtuel)

Prérequis : Python 3.10+, Poetry, Git. Dans PowerShell, à la racine du dépôt :

```powershell
poetry config virtualenvs.in-project true   # une seule fois : .venv dans le projet
poetry install                              # crée .venv et installe tout
.\.venv\Scripts\Activate.ps1                # active l'environnement : "(.venv)" s'affiche
```

- Si PowerShell refuse le script :
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Dans les commandes ci-dessous, `poetry run` marche que l'environnement soit activé
  ou non. Ne le retire pas si tu n'es pas sûr.
- VS Code : `Ctrl+Maj+P` → `Python: Select Interpreter` → choisir `.venv`.
- Mets `churn.csv` dans `data/raw/`.

---

## 3. Lancer chaque partie

DVC et pre-commit exigent un dépôt Git : `git init` si ce n'est pas un clone.

**Données et pipeline DVC**
```powershell
poetry run dvc init                       # une seule fois
poetry run dvc add data/raw/churn.csv     # versionne le dataset complet
poetry run dvc repro                      # lance split puis preprocess
poetry run dvc status                     # attendu : "Data and pipelines are up to date."
```
Attendu : `Reference : 5000 lignes`, `Production : 5000 lignes`, `Train : (4000, 12)`,
`Test : (1000, 12)`. Si tu modifies `reference_size` dans la config, `dvc repro`
relance les deux étapes.

**Drift et dashboard**
```powershell
poetry run python -m src.monitoring.dashboard --scenario none     # drift_share 0.000, critique False
poetry run python -m src.monitoring.dashboard --scenario strong   # drift_share 0.375, critique True
```
Le scénario `strong` doit désigner `Age`, `Balance` et `Geography`. Ouvre
`reports\drift_report.html` dans un navigateur.

**Tests et qualité**
```powershell
poetry run pytest tests -q --cov=src                 # 79 passed, environ 95 %
poetry run flake8 src tests --max-line-length=88     # aucune sortie = OK
git add .
poetry run pre-commit install
poetry run pre-commit run --all-files                # 7 Passed
```

---

## 4. Remote DVC (pour partager les données)

`dvc repro` marche sans remote. Le remote sert à deux choses : les coéquipiers
récupèrent les données, et le prof note « remote storage configuré » (1,5 pt).

**Remote du dépôt : S3** *(déjà déclaré dans `.dvc/config`, non testé ici : il demande
des identifiants AWS)*
```ini
['remote "storage"']
    url = s3://mlops-project-ftima
    profile = dvc
```
Le plugin S3 est dans `pyproject.toml` (`dvc[s3]`). Pour `dvc push` et `dvc pull`, il
faut des identifiants AWS qui ont accès au bucket (à demander à son propriétaire),
enregistrés sous le profil `dvc` :
```powershell
aws configure --profile dvc
poetry run dvc push
```
Sans AWS CLI, tu peux les donner à DVC directement (ils vont dans `.dvc/config.local`,
ignoré par Git : ne les committe jamais) :
```powershell
poetry run dvc remote modify --local storage access_key_id "<clé>"
poetry run dvc remote modify --local storage secret_access_key "<secret>"
```

**Option A : dossier local personnel, si tu n'as pas accès au S3** *(testé sous Linux)*
```powershell
poetry run dvc remote add --local -d local-storage ../dvc-storage
poetry run dvc push
```
`--local` écrit dans `.dvc/config.local` (ignoré par Git) : le remote S3 du dépôt n'est
pas modifié. `../dvc-storage` est un dossier **à côté** du dépôt, jamais dedans.

Un dossier local n'est pas partagé : après un clone, `dvc pull` échoue
(`Is your cache up to date?`). Le chemin à suivre est :
```powershell
git clone https://github.com/kaouech-rahim/Bank-Churn-MLOps.git
cd Bank-Churn-MLOps
poetry config virtualenvs.in-project true
poetry install
.\.venv\Scripts\Activate.ps1
# copier le MEME churn.csv (celui d'Amir) dans data\raw\
poetry run dvc repro
poetry run pytest tests -q
```
Le `churn.csv` doit être strictement identique : compare
`Get-FileHash -Algorithm MD5 data\raw\churn.csv` avec la ligne `md5:` de
`data\raw\churn.csv.dvc`.

**Option B : DagsHub, sans application Google** *(commandes de la doc DagsHub ;
non testé ici, et la doc trouvée vient de copies : vérifie qu'elle correspond à
l'interface actuelle)*
1. Crée un compte DagsHub et un dépôt, invite les 2 autres.
2. Chacun crée un token dans ses paramètres DagsHub.
3. Chacun lance (les 3 dernières lignes vont dans `.dvc/config.local`, ignoré par
   Git : ne partage jamais le token) :
```powershell
poetry run dvc remote add -d origin https://dagshub.com/<utilisateur>/<depot>.dvc
poetry run dvc remote modify origin --local auth basic
poetry run dvc remote modify origin --local user <utilisateur>
poetry run dvc remote modify origin --local password <token>
poetry run dvc push -r origin
```

**Option C : Google Drive, fragile** *(non testé ici)*. La doc DVC indique que
l'application Google par défaut de DVC est bloquée (« This app is blocked »). Il faut
créer ta propre application Google Cloud (Drive API activée, écran de consentement,
identifiant OAuth de type « Application de bureau »), comme décrit dans la doc DVC
« Google Drive ». Des utilisateurs signalent des échecs même ainsi : application « non
vérifiée » (ajoute les 3 adresses comme utilisateurs de test), erreur
`Failed to authenticate GDrive: 'access_token'`, jeton qui expire après 7 jours.
```powershell
poetry add "dvc[gdrive]"
poetry run dvc remote add -d storage gdrive://<ID_du_dossier>
poetry run dvc remote modify --local storage gdrive_client_id "<client-id>"
poetry run dvc remote modify --local storage gdrive_client_secret "<client-secret>"
poetry run dvc push
```
Le dossier Drive doit être partagé avec des personnes précises (« Éditeur »), pas avec
« Tous les utilisateurs disposant du lien ».

**Côté coéquipier :** `git pull`, `poetry install`, configurer ses identifiants avec
les commandes `--local` de l'option choisie, puis `poetry run dvc pull`.

---

## 5. Pour AbdRahim et Fatma

- **Entraînement :** `data/processed/train.csv` et `test.csv` (cible `Exited`).
- **API :** recharger `data/processed/scaler_params.json` pour normaliser une
  requête exactement comme au train ; un batch hors de la plage du train sort de
  [0, 1], c'est voulu.
- **Drift :**
```python
from src.config import load_config
from src.monitoring.drift import detect_drift

report = detect_drift(reference_df, current_df, load_config())
report["drift_share"]       # 0.375
report["critical"]          # True si drift_share >= 0.30  -> lancer le réentraînement
report["drifted_features"]  # ["Age", "Balance", "Geography"]
report["features"]["Age"]   # {"test": "ks", "statistic": ..., "p_value": ..., "psi": ..., "drifted": True}
```
Fatma peut exporter `drift_share` et le `psi` de chaque variable vers Prometheus.

---

## 6. Si ça bloque (Windows)

| Message | Solution |
|---|---|
| `flake8`, `pre-commit` ou `--cov` introuvable | `poetry install` (installe aussi le groupe dev) |
| `&&` n'est pas un séparateur valide | Utilise `;` ou une commande par ligne |
| `is not a git repository` | `git init` |
| pre-commit : `no files to check` | `git add .` puis relance |
| pre-commit : `files were modified` | Normal : `git add .` puis relance jusqu'à 7 Passed |
| Hooks qui bouclent (fins de ligne) | `git config --global core.autocrlf input` |

---

## 7. Vérifications effectuées

79 tests (95 %), 11 bugs injectés volontairement et tous détectés, black / isort /
flake8 / mypy sans erreur, 7 hooks pre-commit passés. Un clone propre avec
`poetry install`, `dvc pull` puis `pytest` fonctionne, et `dvc repro` redonne des
fichiers identiques. Sur 20 découpages et 3 tailles de batch, il n'y a eu aucune
fausse alerte sans drift, et le drift simulé a toujours été détecté.
