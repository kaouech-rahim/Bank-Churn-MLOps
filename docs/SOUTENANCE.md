# Préparation de la soutenance (20 min : 10 pitch + 5 démo + 5 questions)

## 1. Plan du pitch — 10 minutes, 3 orateurs

| Temps | Orateur | Contenu | Support |
|---|---|---|---|
| 0:00–1:00 | Amir | Problème métier : coût d'un départ client, décision de fidélisation, KPI | 1 slide |
| 1:00–2:30 | Amir | Données : 10 000 clients, split 5 000 / 5 000 imposé, schéma, pipeline DVC `split → preprocess → train` | schéma DVC |
| 2:30–4:30 | AbdRahim | 6 runs MLflow, choix du champion (ROC-AUC), Registry Staging → Production, résultats honnêtes (rappel 0,696) | figure comparaison + matrice de confusion |
| 4:30–6:00 | Fatma | API FastAPI (Pydantic, 422, 503), Dockerfile multi-stage non-root, latence p95 47 ms | capture Swagger |
| 6:00–7:00 | Fatma | CI GitHub Actions : lint, pytest ≥ 75 %, build Docker, rapport CML sur PR | capture Actions + commentaire CML |
| 7:00–8:30 | AbdRahim | Prefect : 2 DAG, retries, work pool, CRON ; boucle de réentraînement et règle de promotion | capture DAG |
| 8:30–9:30 | Amir + Fatma | Drift : KS/χ²/PSI, scénario strong (0,375), dashboard Evidently, Prometheus + alerte, Blue/Green | figure PSI |
| 9:30–10:00 | tous | Limites et perspectives | 1 slide |

## 2. Questions probables et réponses courtes

**1. Pourquoi seulement 5 000 lignes pour entraîner ?**
C'est la consigne du professeur : les 5 000 autres servent de flux de production pour
simuler le drift et le réentraînement. Le découpage est stratifié (même taux de churn,
≈ 20,4 %) et reproductible (`random_state = 42`, étape DVC `split`).

**2. Pourquoi choisir le champion sur le ROC-AUC et pas sur le F1 ?**
Le ROC-AUC mesure la qualité du classement indépendamment du seuil ; le seuil reste un
réglage métier séparé. La règle est écrite dans `configs/config.yaml`
(`models.champion_metric`) et testée (`test_select_champion_*`).

**3. Votre rappel est 0,696 alors que l'objectif est 0,70. Pourquoi ne pas baisser le seuil ?**
Au seuil 0,45, le rappel monte à 0,75 sur le test, mais choisir le seuil sur le jeu de test
reviendrait à tricher sur l'évaluation. Il faudrait un jeu de validation dédié : c'est une
perspective. Nous avons préféré rapporter le résultat tel quel.

**4. Comment évitez-vous la fuite de données dans le prétraitement ?**
Le min-max est appris sur le train uniquement (`fit_scaler_params`) puis appliqué au test,
à la production et à l'API avec les mêmes paramètres (`scaler_params.json` livré avec le modèle).

**5. Comment l'API applique-t-elle le même prétraitement qu'à l'entraînement ?**
`ModelService` appelle la fonction `encode_features` du module de prétraitement : aucun
code dupliqué. Un test vérifie que la probabilité de l'API est identique à celle du
modèle appliqué à `encode_features`.

**6. Que se passe-t-il si le modèle est absent au démarrage du conteneur ?**
L'API démarre, mais `/health` renvoie 503 : la sonde Docker marque le conteneur
*unhealthy* et le script Blue/Green refuse de basculer vers lui.

**7. Pourquoi un test ET le PSI pour déclarer une variable en drift ?**
Sur des milliers de lignes, KS ou χ² détectent des écarts minuscules. Exemple réel :
`EstimatedSalary` a p = 0,003 mais un PSI de 0,013 ; elle n'est pas comptée en drift.

**8. Comment est calculé le PSI ?**
Σ (p_cur − p_ref) × ln(p_cur / p_ref) sur 10 tranches définies par les quantiles de la
référence (une tranche par modalité pour les variables catégorielles), avec un epsilon
pour éviter ln(0). Un test compare la valeur à un calcul fait à la main.

**9. Sur quelles données réentraînez-vous ? Les labels ne sont-ils pas inventés ?**
Référence + une partie du lot de production. Les labels sont les vrais labels du pool ;
en revanche le drift est simulé sur les variables, donc ces labels sont une approximation.
Si le lot n'a pas de labels, le flow saute le réentraînement (testé).

**10. Comment décidez-vous de promouvoir le nouveau modèle ?**
Le lot de production est coupé en deux parties disjointes avant la simulation. Les deux
modèles sont évalués sur le *hold-out* qu'aucun n'a vu ; le nouveau est promu seulement
si son ROC-AUC est ≥ à celui de l'actuel. Sur nos données : 0,8457 contre 0,8067 → promu.

**11. Pourquoi avez-vous changé cette règle en cours de projet ?**
Notre première version comparait les modèles sur un test qui contenait des lignes de
référence déjà vues par l'ancien modèle : il obtenait 0,9115, un score gonflé. Nous avons
corrigé le biais avec le hold-out disjoint.

**12. Qu'apporte le Blue/Green ?**
Le nouveau modèle est chargé sur la couleur inactive, contrôlé (`/health` + version via
`/info`), puis Nginx est rechargé gracieusement. L'ancienne couleur reste démarrée :
`--rollback` revient en arrière en une commande. (À démontrer avec Docker.)

**13. Pourquoi la CI n'utilise-t-elle pas le vrai dataset ?**
La CI ne doit dépendre ni de S3 ni de secrets. Les tests et le rapport CML utilisent des
données synthétiques de même schéma ; les vraies métriques viennent de `dvc repro` et MLflow.

**14. Les stages Staging/Production sont dépréciés dans MLflow 3, pourquoi les utiliser ?**
Le TP3 et le guide demandent la promotion formelle vers *Production* ; l'API MLflow 3.17
les supporte encore. Nous posons en plus l'alias `champion` pour les versions futures.

**15. Comment prouvez-vous que vos tests vérifient vraiment la logique ?**
Couverture ≥ 75 % exigée en CI, et tests de mutation manuels : nous avons volontairement
cassé le choix du champion, la règle de promotion, les seuils de risque et le
déclenchement du réentraînement ; à chaque fois des tests ont échoué (voir le rapport).
