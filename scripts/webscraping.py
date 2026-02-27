# Chargement des bibliothèques
import requests
import pandas as pd
from time import sleep

# Ce script récupère des indicateurs de la Banque Mondiale (World Bank API)
# pour tous les pays entre 2010 et 2025, en gérant la pagination, les timeouts,
# puis :
#  - filtre les "pays réels",
#  - exporte les données au format long,
#  - transforme ensuite en format wide (ligne = (iso3, pays, année),
#    colonnes = indicateurs).

# Téléchargement d'un indicateur WB pour tous les pays de 2010 à 2025
def wb_indicator_all(indicator, start=2010, end=2025, verbose=True):

    # Liste de toutes les lignes extraites
    rows = []
    page = 1

    # Paramètres de requête
    PER_PAGE = 5000      # maximum d'observations par page
    TIMEOUT = (10, 180) 
    MAX_RETRIES = 5    # nombre maximum d'essai si ReadTimeout

    if verbose:
        print(f"\nIndicateurs: {indicator} | {start}-{end}")

    # Boucle de pagination : on parcourt page=1..total_pages
    while True:
        # URL API pour l'indicateur
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": PER_PAGE,
            "page": page,
            "date": f"{start}:{end}"
        }

        # Relance en cas de ReadTimeout
        attempt = 1
        while True:
            try:
                r = requests.get(url, params=params, timeout=TIMEOUT)
                break
            except requests.exceptions.ReadTimeout:
                # Si on a atteint le maximum d'essai, on remonte l'erreur
                if attempt >= MAX_RETRIES:
                    raise
                if verbose:
                    print(f"Timeout page {page} (tentative {attempt}/{MAX_RETRIES})")
                attempt += 1
                # backoff simple (augmente l'attente à chaque tentative)
                sleep(1.5 * attempt)

        # Si HTTP != 200, on stoppe avec une erreur
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} sur indicateur {indicator} (page {page})")

        # Décodage JSON
        data = r.json()

        # Si l'API ne renvoie rien
        if not data or len(data) < 2 or data[1] is None:
            if verbose:
                print("Aucune donnée retournée.")
            break

        # Extraction des observations
        for obs in data[1]:
            # ISO3 officiel s'il existe
            iso3 = obs.get("countryiso3code")

            # Identifiant brut du pays côté WB
            iso2_or_code = (obs.get("country") or {}).get("id")

            # Nom du pays
            country_name = (obs.get("country") or {}).get("value")

            # Année + valeur
            year = obs.get("date")
            value = obs.get("value")

            # On garde uniquement si iso3 et année existent
            if iso3 and year:
                rows.append({
                    "iso3": iso3,
                    "country": country_name,
                    "year": int(year),
                    "indicator": indicator,
                    "value": value,
                    "country_id_raw": iso2_or_code
                })

        # Nombre total de pages
        total_pages = data[0].get("pages", page)

        if verbose:
            print(f"Page {page} sur {total_pages} | nombre de lignes : {len(rows)}")

        # Sortie de boucle si on a atteint la dernière page
        if page >= total_pages:
            break

        page += 1
        sleep(0.25)

    # Conversion finale en DataFrame
    return pd.DataFrame(rows)

# Liste ISO3 des pays réels
# L'API World Bank contient aussi des agrégats, donc on filtre en excluant ceux dont region.id == "NA".
def wb_real_countries(verbose=True):

    countries = []
    page = 1

    while True:
        url = "https://api.worldbank.org/v2/country"
        params = {"format": "json", "per_page": 400, "page": page}
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} sur /country (page {page})")

        data = r.json()
        if not data or len(data) < 2 or data[1] is None:
            break

        # Parcours des pays renvoyés
        for c in data[1]:
            iso3 = c.get("id")  # ici c'est bien ISO3
            region_id = (c.get("region") or {}).get("id")

            # On garde uniquement les codes à 3 lettres et dont region_id != NA
            if iso3 and len(iso3) == 3 and region_id != "NA":
                countries.append(iso3)

        total_pages = data[0].get("pages", page)
        if verbose:
            print(f"country page {page}/{total_pages} | pays réels cumulés: {len(countries)}")

        if page >= total_pages:
            break

        page += 1
        sleep(0.1)

    return sorted(set(countries))

# Téléchargement en deux temps : 2010-2017 et 2018-2025 + filtrage des pays réels
def wb_indicator_all_split(indicator, verbose=True):

    # Récupération de la liste des pays réels (ISO3)
    real_iso3 = set(wb_real_countries(verbose=False))

    # Téléchargement sur 2 périodes
    df1 = wb_indicator_all(indicator, start=2010, end=2017, verbose=verbose)
    df2 = wb_indicator_all(indicator, start=2018, end=2025, verbose=verbose)

    # Fusion des deux périodes
    df = pd.concat([df1, df2], ignore_index=True)

    # Filtre : on ne garde que les pays réels
    df = df[df["iso3"].isin(real_iso3)].copy()

    if verbose:
        print(f"Pays réels : {df['iso3'].nunique()} pays | {len(df)} lignes")
    return df

# Extraction multi-indicateurs
if __name__ == "__main__":
    # Liste d'indicateurs WB utiles
    indicators = [
        "SP.POP.TOTL",  # Population totale
        "NY.GDP.PCAP.CD",  # PIB par habitant (USD courant)
        "SH.XPD.CHEX.GD.ZS",  # Dépenses santé (% PIB)
        "SP.URB.TOTL.IN.ZS"  # Urbanisation (%)
    ]

    # Dictionnaire pour renommer les colonnes
    rename_map = {
        "SP.POP.TOTL": "population",
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "SH.XPD.CHEX.GD.ZS": "health_exp_gdp",
        "SP.URB.TOTL.IN.ZS": "urban_pop_pct"
    }

    all_long = []

    # Extraction indicateur par indicateur (format long)
    for ind in indicators:
        df_ind = wb_indicator_all_split(ind, verbose=True)
        all_long.append(df_ind)

    # Concaténation au format long
    df_long = pd.concat(all_long, ignore_index=True)

    # Sauvegarde du dataset obtenu au format long
    df_long.to_csv("wb_data_long_2010_2025.csv", index=False)

    # Passage au format wide :
    # une ligne = (iso3, country, year)
    # colonnes = indicateurs
    df_wide = (
        df_long.pivot_table(
            index=["iso3", "country", "year"],
            columns="indicator",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # Renommer les colonnes indicateurs
    df_wide = df_wide.rename(columns=rename_map)

    # Sauvegarde du dataset wide
    df_wide.to_csv("wb_data_wide_2010_2025.csv", index=False)

    # Vérification rapide
    print(df_wide.head(10))