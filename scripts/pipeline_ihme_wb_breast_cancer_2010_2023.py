# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import pycountry
import pycountry_convert as pc
import unicodedata
import re
from difflib import get_close_matches
from pathlib import Path


# ===== SELECTION DES COLONNES UTILES DU DATASET IHME-GBD =====

# Chargement du dataset IHME-GBD
df = pd.read_csv("IHME-GBD_2023_DATA.csv")

# Filtres de base
df = df[
    (df["sex_name"] == "Femme") &
    (df["population_group_name"] == "Toute la population") &
    (df["age_name"].str.contains("Normalisé", case=False, na=False)) &
    (df["cause_name"] == "Cancer du sein") &
    (df["year"] >= 2010) &
    (df["year"] <= 2023)
]

# Incidence et mortalité (ASR uniquement)
incidence = df[
    (df["measure_name"] == "Incidence") &
    (df["metric_name"] == "Taux")
][["location_id", "location_name", "year", "val"]].rename(
    columns={"val": "incidence_asr"}
)

mortality = df[
    (df["measure_name"] == "Décès") &
    (df["metric_name"] == "Taux")
][["location_id", "location_name", "year", "val"]].rename(
    columns={"val": "mortality_asr"}
)

# Fusion de l'incidence et mortalité
ihme_final = incidence.merge(
    mortality,
    on=["location_id", "location_name", "year"],
    how="left"
)

# Vérification
print(ihme_final.head())

# Sauvegarde
ihme_final.to_csv(
    "ihme_breast_cancer_asr_2010_2023.csv",
    index=False
)


# ===== NETTOYAGE ET HARMONISATION DES DONNEES IHME =====

# Chemin du fichier IHME
path = Path("ihme_breast_cancer_asr_2010_2023.csv")

# On récupère le contenu brut du fichier sans supposer l'encodage
raw = path.read_bytes()

# On tente plusieurs décodages courants
encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]

# Fonction de "score" pour détecter de mauvais encodages
def score(text: str) -> int:
    bad = 0
    bad += text.count("�") * 50 
    bad += text.count("Ã") * 10 
    bad += text.count("â") * 5 
    return bad

# On teste chaque encodage et on garde un score pour déterminer lequel sera meilleur
candidates = []
for enc in encodings:
    try:
        txt = raw.decode(enc, errors="replace")
        candidates.append((score(txt), enc, txt[:300]))
    except Exception:
        pass

# On fait le tri de décodage selon le score obtenu
candidates.sort(key=lambda x: x[0])

# Affiche les meilleurs candidats pour validation rapide
print("Meilleurs décodages:")
for s, enc, preview in candidates[:3]:
    print(f"- {enc} | score={s} | aperçu={preview.replace('\\n',' ')[:120]}")

# On choisit l'encodage avec un score égal à 0
best_enc = candidates[0][1]

# Décodage final du fichier brut avec l'encodage choisi
text = raw.decode(best_enc, errors="replace")

# Sauvegarde du texte décodé dans un fichier utf-8 pour éviter les problèmes d'encodage ensuite
clean_path = path.with_suffix(".clean_utf8.csv")
clean_path.write_text(text, encoding="utf-8", newline="")

# Lecture du CSV nettoyé (UTF-8) avec pandas
ihme = pd.read_csv(clean_path)

# Contrôle rapide : dimensions + aperçu
print(ihme.shape)
print(ihme.head())

# Normalisation de chaînes (pays) pour robustifier le mapping
def norm(s: str) -> str:
    # Conversion en string, suppression des espaces aux bords et conversion en minuscules
    s = str(s).strip().lower()

    # Normalisation de certains caractères typographiques (apostrophes/tirets)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("-", " ")

    # Suppression de tout contenu entre parenthèses
    s = re.sub(r"\(.*?\)", " ", s)

    # Suppression des accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # Maintien uniquement des lettres/chiffres/espaces
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)

    # Normalisation des espaces multiples
    s = re.sub(r"\s+", " ", s).strip()

    # Suppression de certains articles ou préfixes fréquents au début
    for prefix in ("l'", "d'", "l ", "la ", "le ", "les ", "de ", "des ", "du "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    return s

# Mapping manuel FR : EN pour les noms de pays fréquents en français afin de reconnaître les noms FR
FR2EN = {
    "libye": "libya",
    "maroc": "morocco",
    "albanie": "albania",
    "suisse": "switzerland",
    "barbade": "barbados",
    "japon": "japan",
    "equateur": "ecuador",
    "allemagne": "germany",
    "royaume uni": "united kingdom",
    "ouganda": "uganda",
    "comores": "comoros",
    "zambie": "zambia",
    "inde": "india",
    "bosnie herzegovine": "bosnia and herzegovina",
    "iles salomon": "solomon islands",
    "bolivie": "bolivia",
    "bresil": "brazil",
    "tanzanie": "tanzania",
    "chine": "china",
    "la chine": "china",
    "coree du nord": "korea, democratic people's republic of",
    "la coree du nord": "korea, democratic people's republic of",
    "coree du sud": "korea, republic of",
    "arabie saoudite": "saudi arabia",
    "cap vert": "cabo verde",
    "cap-vert": "cabo verde",
    "sao tome et principe": "sao tome and principe",
    "croatie": "croatia",
    "bulgarie": "bulgaria",
    "cameroun": "cameroon",
    "slovaquie": "slovakia",
    "slovenie": "slovenia",
    "estonie": "estonia",
    "lettonie": "latvia",
    "dominique": "dominica",
    "palestine": "palestine, state of",

    "royaume-uni": "united kingdom",
    "pays bas": "netherlands",
    "pays-bas": "netherlands",
    "republique tcheque": "czechia",
    "republique tcheque": "czechia",
    "italie": "italy",
    "autriche": "austria",
    "suede": "sweden",
    "islande": "iceland",
    "hongrie": "hungary",
    "russie": "russian federation",
    "kirghizistan": "kyrgyzstan",
    "nouvelle zelande": "new zealand",
    "nouvelle-zelande": "new zealand",
    "emirats arabes unis": "united arab emirates",
    "emirats-arabes-unis": "united arab emirates",

    "republique democratique du congo": "congo, the democratic republic of the",
    "republique democratique du congo": "congo, the democratic republic of the",

    "republique dominicaine": "dominican republic",
    "antigua et barbuda": "antigua and barbuda",
    "antigua-et-barbuda": "antigua and barbuda",

    "iles mariannes du nord": "northern mariana islands",
    "iles mariannes du nord": "northern mariana islands",

    "cambodge": "cambodia",
    "irak": "iraq",
    "la guinee equatoriale": "equatorial guinea",
    "brunei": "brunei darussalam",
    "norvege": "norway",
    "pologne": "poland",
    "afrique du sud": "south africa",
    "macedoine": "north macedonia",
    "chypre": "cyprus",
    "iles marshall": "marshall islands",
    "papouasie nouvelle guinee": "papua new guinea",
    "republique centrafricaine": "central african republic",
    "mexique": "mexico",
    "swaziland": "eswatini",
    "etats unis": "united states",
    "trinite et tobago": "trinidad and tobago",
    "samoa americaines": "american samoa",
    "espagne": "spain",
    "ile maurice": "mauritius",
    "turquie": "turkiye",
    "etats federes de micronesie": "micronesia, federated states of",
    "sud soudan": "south sudan",
    "jamaique": "jamaica",
    "koweit": "kuwait",
    "malaisie": "malaysia",
    "sainte lucie": "saint lucia",
    "iles de cook": "cook islands",
    "saint christophe et nieves": "saint kitts and nevis",
    "guinee equatoriale": "equatorial guinea",

    "erythree": "eritrea",
}

# Construction d'un index pycountry : nom normalisé ISO3
# On construit un dictionnaire de correspondance à partir de pycountry en utilisant plusieurs attributs potentiels
index = {}
for c in pycountry.countries:
    for attr in ("name", "official_name", "common_name"):
        v = getattr(c, attr, None)
        if v:
            index[norm(v)] = c.alpha_3

# Liste des clés normalisées
keys = list(index.keys())

# Fonction de conversion de nom de pays  code ISO3
def to_iso3(name: str):
    # Normalisalisation du nom entré
    n = norm(name)

    # exceptions pour erythree
    if n == "erythree":
        return "ERI"

    # Si le nom FR est connu, on mappe d'abord vers EN et on cherche dans l'index
    if n in FR2EN:
        n2 = norm(FR2EN[n])
        if n2 in index:
            return index[n2]

    # Si le nom normalisé correspond exactement à pycountry
    if n in index:
        return index[n]

    # Sinon, tentative de correspondance approchée sur les clés connues
    m = get_close_matches(n, keys, n=1, cutoff=0.80)
    if m:
        return index[m[0]]

    # Si tout échoue, on renvoie None
    return None

# Application du mapping sur la colonne IHME contenant les noms de pays
ihme["iso3"] = ihme["location_name"].apply(to_iso3)

# Diagnostics: combien de pays uniques et combien mappés
print("Pays IHME :", ihme["location_name"].nunique())
print("Pays mappés ISO3 :", ihme["iso3"].nunique())

# Liste des pays non mappés (top 50) pour inspection
non = ihme[ihme["iso3"].isna()]["location_name"].value_counts().head(50)
print(non)

# Sauvegarde finale avec la colonne ISO3 ajoutée
ihme.to_csv(
    "ihme_breast_cancer_asr_2010_2023_iso3.csv",
    index=False
)

# ===== FUSION DES DATASETS ET AJOUT D'UNE COLONNE : continents =====

# Chargement des deux datasets IHME et Banque Mondiale
ihme = pd.read_csv("ihme_breast_cancer_asr_2010_2025_iso3.csv")
wb = pd.read_csv("wb_data_wide_2010_2025.csv")

# Contrôle des dimensions
print("IHME :", ihme.shape)
print("WB   :", wb.shape)

# Fusion IHME et WB sur iso3 et year
# Fusion gauche : on garde toutes les lignes IHME, et on ajoute les variables WB
dataset = ihme.merge(
    wb,
    on=["iso3", "year"],
    how="left"
)

# Contrôle de la taille après fusion
print("Dataset fusionné :", dataset.shape)

# Ajout d'une variable continents
def iso3_to_continent(iso3):
    # Si iso3 est manquant
    if pd.isna(iso3):
        return np.nan

    # Standardisation de iso3
    iso3 = str(iso3).strip().upper()

    # Ajout des exceptions fréquentes
    special = {
        "XKX": "Europe",
        "TLS": "Asie",
    }
    if iso3 in special:
        return special[iso3]

    try:
        # Conversion ISO3 en ISO2
        country_alpha2 = pc.country_alpha3_to_country_alpha2(iso3)

        # ISO2 : code continent (AF, EU, AS, NA, SA, OC)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)

        # Mapping du code continent
        continent_map = {
            "AF": "Afrique",
            "EU": "Europe",
            "AS": "Asie",
            "NA": "Amérique du Nord",
            "SA": "Amérique du Sud",
            "OC": "Océanie"
        }
        return continent_map.get(continent_code, np.nan)

    # Si un iso3 ne peut pas être converti, on renvoie NaN
    except Exception:
        return np.nan

# Nettoyage/normalisation de la colonne iso3 avant mapping continent
dataset["iso3"] = dataset["iso3"].astype(str).str.strip().str.upper()

# Remplacement de certaines valeurs "fausses" issues de conversions/lecture CSV en NaN
dataset.loc[dataset["iso3"].isin(["NAN", "NONE", ""]), "iso3"] = np.nan

# Création de la colonne continent standardisée
dataset["region_std"] = dataset["iso3"].apply(iso3_to_continent)

# Vérification des continents et iso3 non mappés
print("Continents créés :", dataset["region_std"].value_counts(dropna=False))
print("ISO3 non mappés :", dataset.loc[dataset["region_std"].isna(), "iso3"].value_counts().head(20))

# Export du dataset final fusionné
dataset.to_csv(
    "dataset_final_cancer_sein_IHME_WB_2010_2023.csv",
    index=False
)