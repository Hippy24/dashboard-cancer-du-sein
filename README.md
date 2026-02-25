# **Tableau de bord épidémiologique mondial du cancer de sein : 2010 - 2023**

Le cancer du sein constitue l’un des cancers les plus fréquents chez les femmes à l’échelle mondiale. Son incidence et sa mortalité varient fortement selon le niveau de développement économique, l’accès au dépistage, la qualité du système de santé, la structure démographique et l’urbanisation.

L’objectif de ce projet est de construire un système d’analyse épidémiologique, allant de la collecte des données jusqu’à un tableau de bord interactif, permettant :
- l’analyse descriptive,
- l’exploration des disparités internationales,
- l’analyse temporelle,
- la modélisation statistique,
- la stratification du risque,
- le regroupement de pays selon leur profil.

## 1. Collecte de données
### 1.1. Données provenant de l'IHME - GBD (Global Burden Desease)
Dans un premier temps, des données issues du programme Global Burden of Disease (GBD) de l’IHME ont été téléchargées selon les indicateurs suivants :
- Cause : Cancer du sein
- Sexe : Femmes
- Âge : Standardisé selon l’âge
- Mesures sélectionnées : Incidence (ASR) et Mortalité (ASR)
- Période : 2010–2023
- Niveau : Pays
  
Les colonnes retenues sont les suivantes :
- location_id,
- location_name,
- year,
- incidence_asr,
- mortality_asr.

### 1.2. Données de la Banque Mondiale : Webscraping
Ensuite, les indicateurs suivants ont été extraits pour chaque pays et chaque année grâce à l’API Banque Mondiale :
- PIB par habitant (gdp_per_capita),
- Dépenses de santé (% du PIB), (health_exp_gdp),
- Population totale (population),
- Population urbaine (urban_pop_pct).

Ces indicateurs ont été utilisés pour rendre les modèles statistiques plus riches et réaliser le clustering des pays.

## 2. Nettoyage et harmonisation des données
### 2.1. Rectification de l'encodage
Les noms de certains pays ont montré des problèmes d'encodage. Nous avons donc réalisé :
- le réencodage en utf-8,
- le nettoyage des caractères spéciaux,
- l'uniformisation des apostrophes et accents.

### 2.2. Harmonisation des pays
Les noms issus de l'IHME ne correspondent pas toujours exactement à ceux de la Banque Mondiale. Donc, nous avons effectué :
- la suppression des espaces superflus,
- la correction des variantes (exemple : États Unis/United States),
- le mapping manuel pour les cas complexes,
- la conversion vers le code ISO3 avec (pycountry).

*Au total, **203 pays IHME** ont été mappés.*

## 3. Jointure des bases de données
Après l'obtention des différentes bases de données, la jointures entre elles a été faites sur les clés :
- iso3,
- année (year).
  
La base de données finale obtenue est composée des variables suivantes :
- location_id,
- location_name,
- incidence_asr,
- mortality_asr,
- population,
- gdp_per_capita,
- health_exp_gdp,
- urban_pop_pct,
- iso3,
- country,
- region_std (continent).

## 4. Construction des indicateurs épidémiologiques
Cette étape est réalisée dans la fonction **add_features()** du code.

### 4.1. Estimation des cas et décès
Nous avons estimé les cas et décès suivant la formule :
- estimated_cases = (incidence_asr / 100000) * population
- estimated_deaths = (mortality_asr / 100000) * population

Ces estimations permettent une lecture plus concrète des volumes de cas et décès.

### 4.2. Détermination de la létalité (Ratio mortalité / incidence)

Elle a été calculée selon la formule : fatality_proxy = mortality_asr / incidence_asr

Ce ratio permet d’apprécier la sévérité relative du cancer.

### 4.3. Variations annuelles
Nous avons calculé les variations annuelles par pays (pct_change() × 100).

Elles permettent d’identifier :

- accélérations,
- ralentissements,
- ruptures de tendance.



















