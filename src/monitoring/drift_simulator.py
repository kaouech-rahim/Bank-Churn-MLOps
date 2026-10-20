"""Simulateur de data drift : fabrique un flux de production qui "derive"."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DriftSimulator:
    """Injecte un drift controle dans un DataFrame de production.

    Un scenario est un dictionnaire de parametres (defini dans config.yaml) :
        - age_shift : decalage moyen de l'age (ex : 8 = +8 ans)
        - balance_factor : multiplicateur des soldes strictement positifs
        - target_country + target_share : nouvelle part d'un pays dans la clientele
    Le scenario "none" (dictionnaire vide) renvoie les donnees inchangees.
    """

    def __init__(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        age_bounds: Tuple[int, int] = (18, 100),
        age_column: str = "Age",
        balance_column: str = "Balance",
        country_column: str = "Geography",
    ) -> None:
        """Initialise le simulateur.

        Args:
            scenarios: {nom_du_scenario: parametres}.
            age_bounds: Bornes (min, max) appliquees a l'age apres decalage.
            age_column: Nom de la colonne age.
            balance_column: Nom de la colonne solde.
            country_column: Nom de la colonne pays.
        """
        self.scenarios = scenarios
        self.age_bounds = age_bounds
        self.age_column = age_column
        self.balance_column = balance_column
        self.country_column = country_column

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DriftSimulator":
        """Construit le simulateur a partir de la configuration du projet.

        Args:
            cfg: Configuration complete (sections drift et schema).

        Returns:
            DriftSimulator: Simulateur pret a l'emploi.
        """
        low, high = cfg["schema"]["ranges"]["Age"]
        return cls(cfg["drift"]["scenarios"], age_bounds=(int(low), int(high)))

    def apply(
        self, df: pd.DataFrame, scenario: str = "strong", seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """Applique un scenario de drift et renvoie une copie modifiee.

        Args:
            df: DataFrame de production (non modifie).
            scenario: Nom du scenario a appliquer.
            seed: Graine aleatoire (meme graine = meme resultat).

        Returns:
            pd.DataFrame: Copie du DataFrame avec le drift injecte.

        Raises:
            ValueError: Si le scenario est inconnu ou ses parametres invalides.
        """
        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario inconnu : {scenario} (disponibles : {list(self.scenarios)})"
            )

        params = self.scenarios[scenario]
        data = df.copy()  # aucune mutation de l'entree
        rng = np.random.default_rng(seed)

        # 1. Changement de la repartition des pays (re-echantillonnage pondere)
        if "target_share" in params:
            data = self._reweight_country(
                data, params.get("target_country"), params["target_share"], rng
            )

        # 2. Les clients sont plus ages (decalage + petit bruit gaussien)
        if "age_shift" in params:
            noise = rng.normal(0.0, 1.0, len(data))
            aged = (data[self.age_column] + params["age_shift"] + noise).round()
            data[self.age_column] = aged.clip(*self.age_bounds).astype(int)

        # 3. Les soldes augmentent (les comptes a 0 restent a 0)
        if "balance_factor" in params:
            balance = data[self.balance_column]
            data[self.balance_column] = np.where(
                balance > 0, balance * params["balance_factor"], 0.0
            )

        return data.reset_index(drop=True)

    def _reweight_country(
        self,
        df: pd.DataFrame,
        country: Optional[str],
        target_share: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Re-echantillonne (avec remise) pour que `country` atteigne `target_share`.

        Args:
            df: DataFrame de production.
            country: Pays dont on veut changer la part (ex : Germany).
            target_share: Part visee pour ce pays, strictement entre 0 et 1.
            rng: Generateur aleatoire (garantit la reproductibilite).

        Returns:
            pd.DataFrame: DataFrame re-echantillonne, de meme taille que df.

        Raises:
            ValueError: Si country est manquant, si target_share n'est pas dans
                ]0, 1[ ou si la part actuelle du pays vaut 0 ou 1.
        """
        if country is None:
            raise ValueError("target_country est obligatoire avec target_share.")
        if not 0 < target_share < 1:
            raise ValueError(
                f"target_share doit etre dans ]0, 1[ (recu : {target_share})"
            )

        is_country = (df[self.country_column] == country).to_numpy()
        current_share = float(is_country.mean())
        if current_share in (0.0, 1.0):
            raise ValueError(f"Part actuelle de {country} invalide : {current_share}")

        weights = np.where(
            is_country,
            target_share / current_share,
            (1 - target_share) / (1 - current_share),
        )
        probabilities = weights / weights.sum()
        indices = rng.choice(len(df), size=len(df), replace=True, p=probabilities)
        return df.iloc[indices].reset_index(drop=True)
