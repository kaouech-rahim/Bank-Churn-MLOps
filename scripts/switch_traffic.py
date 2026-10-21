"""Bascule Blue/Green du trafic de l'API derriere Nginx.

Deroulement d'une bascule (couleur active = blue, par exemple) :
1. copie du nouveau modele (models/) dans deploy/models/green ;
2. redemarrage du conteneur api_green, qui charge ce modele ;
3. attente que http://localhost:8002/health renvoie 200 ET que /info annonce
   la version attendue (sinon : on abandonne, blue continue de servir) ;
4. reecriture de deploy/nginx/upstream.conf vers api_green puis
   "nginx -s reload" (rechargement gracieux : les requetes en cours finissent) ;
5. blue reste demarre avec l'ancien modele : --rollback rebascule en une commande.

Usage (PowerShell, a la racine du projet) :
    poetry run python scripts/switch_traffic.py --init      # 1er demarrage
    poetry run python scripts/switch_traffic.py             # deployer models/
    poetry run python scripts/switch_traffic.py --rollback  # revenir en arriere
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

# Permet "python scripts/switch_traffic.py" depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402

COLORS = ("blue", "green")
BUNDLE_FILES = ("model.joblib", "metadata.json", "scaler_params.json")
Runner = Callable[[List[str]], Any]
Getter = Callable[..., Any]


def other_color(color: str) -> str:
    """Renvoie l'autre couleur (blue <-> green).

    Raises:
        ValueError: Si la couleur est inconnue.
    """
    if color not in COLORS:
        raise ValueError(f"Couleur inconnue : {color}")
    return "green" if color == "blue" else "blue"


def read_active_color(path: Path) -> str:
    """Lit la couleur active (blue si le fichier est absent ou invalide)."""
    try:
        color = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "blue"
    return color if color in COLORS else "blue"


def render_upstream(color: str) -> str:
    """Contenu du fichier upstream.conf pour une couleur."""
    other_color(color)  # valide la couleur
    return (
        f"# Couleur active : {color} (fichier genere par scripts/switch_traffic.py)\n"
        "upstream churn_api {\n"
        f"    server api_{color}:8000;\n"
        "}\n"
    )


def write_upstream(path: Path, color: str) -> None:
    """Reecrit upstream.conf EN PLACE (meme fichier : le montage Docker le voit)."""
    with open(path, "r+", encoding="utf-8") as f:
        f.seek(0)
        f.write(render_upstream(color))
        f.truncate()


def copy_bundle(source_dir: Path, target_dir: Path) -> None:
    """Copie model.joblib, metadata.json et scaler_params.json.

    Raises:
        FileNotFoundError: Si un fichier du modele manque dans source_dir.
    """
    missing = [f for f in BUNDLE_FILES if not (source_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Fichiers manquants dans {source_dir} : {missing}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        shutil.copyfile(source_dir / name, target_dir / name)


def wait_for_version(
    base_url: str,
    expected_version: Optional[str],
    timeout: float,
    interval: float = 1.0,
    getter: Getter = httpx.get,
) -> bool:
    """Attend que /health renvoie 200 et que /info annonce la version attendue.

    Args:
        base_url: URL de l'instance (ex : http://localhost:8002).
        expected_version: Version attendue (None : seul /health est verifie).
        timeout: Temps maximal d'attente en secondes.
        interval: Pause entre deux essais.
        getter: Fonction HTTP GET (injectable pour les tests).

    Returns:
        bool: True si l'instance est prete dans le delai.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if getter(f"{base_url}/health", timeout=3).status_code == 200:
                if expected_version is None:
                    return True
                info = getter(f"{base_url}/info", timeout=3).json()
                if str(info.get("version")) == str(expected_version):
                    return True
        except httpx.HTTPError:
            pass  # instance pas encore demarree
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def switch_traffic(
    cfg: Dict[str, Any],
    source_dir: Path,
    rollback: bool = False,
    runner: Runner = lambda cmd: subprocess.run(cmd, check=True),
    getter: Getter = httpx.get,
    interval: float = 1.0,
) -> Dict[str, Any]:
    """Deploie le modele de source_dir sur la couleur inactive puis bascule.

    Args:
        cfg: Configuration complete (section deployment).
        source_dir: Dossier contenant le nouveau modele (models/).
        rollback: Si True, rebascule vers l'autre couleur sans copier de modele.
        runner: Execute une commande docker (injectable pour les tests).
        getter: Fonction HTTP GET (injectable pour les tests).
        interval: Pause entre deux verifications de sante.

    Returns:
        Dict[str, Any]: switched, active, previous, reason eventuelle.
    """
    dep = cfg["deployment"]
    active_file = Path(dep["active_color_file"])
    active = read_active_color(active_file)
    target = other_color(active)
    expected: Optional[str] = None

    if not rollback:
        target_dir = Path(dep["models_dir"]) / target
        copy_bundle(source_dir, target_dir)
        with open(target_dir / "metadata.json", "r", encoding="utf-8") as f:
            expected = str(json.load(f).get("version"))
        print(f"[DEPLOY] Modele v{expected} copie dans {target_dir}")
        runner(["docker", "compose", "restart", f"api_{target}"])

    url = dep[f"{target}_url"]
    print(f"[CHECK] Attente de {url}/health ...")
    if not wait_for_version(
        url, expected, dep["health_timeout_seconds"], interval, getter
    ):
        print(f"[ABANDON] {target} n'est pas pret : {active} continue de servir.")
        return {"switched": False, "active": active, "reason": "health_check_failed"}

    write_upstream(Path(dep["nginx_upstream_file"]), target)
    runner(["docker", "compose", "exec", "-T", "nginx", "nginx", "-s", "reload"])
    active_file.write_text(target + "\n", encoding="utf-8")
    print(
        f"[OK] Trafic bascule de {active} vers {target} ({active} garde pour rollback)"
    )
    return {"switched": True, "active": target, "previous": active}


def init_deployment(cfg: Dict[str, Any], source_dir: Path) -> None:
    """Premier demarrage : meme modele sur blue et green, trafic sur blue."""
    dep = cfg["deployment"]
    for color in COLORS:
        copy_bundle(source_dir, Path(dep["models_dir"]) / color)
    write_upstream(Path(dep["nginx_upstream_file"]), "blue")
    Path(dep["active_color_file"]).write_text("blue\n", encoding="utf-8")
    print("[INIT] Modele copie sur blue et green, trafic sur blue.")
    print("       Lancez maintenant : docker compose up -d --build")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bascule Blue/Green")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-dir", default="models", help="Modele a deployer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init", action="store_true", help="Preparer blue et green")
    group.add_argument("--rollback", action="store_true", help="Revenir en arriere")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.init:
        init_deployment(config, Path(args.model_dir))
    else:
        outcome = switch_traffic(config, Path(args.model_dir), rollback=args.rollback)
        sys.exit(0 if outcome["switched"] else 1)
