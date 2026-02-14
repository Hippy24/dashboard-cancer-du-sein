import pandas as pd
import numpy as np

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table


CSV_PATH = "dataset_final_cancer_sein_IHME_WB_2010_2025.csv"

SIDEBAR_BG = "#071A33"
CONTENT_BG = "#BFE3FF"
TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#06203A"

CARD_TREND = "#EAF2FF"
CARD_MAP = "#FFF0F6"

TITLE_CARD_BG = "#FFE1EE"
PINK_SLOGAN = "#FF4FA3"

CASES_COLOR = "#1F77B4"
DEATHS_COLOR = "#D62728"

# (Comparaison Monde supprimée)
MAP_MARK_YELLOW = "#FFD400"
MAP_MARK_SIZE = 4

FA_CSS = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"

KPI_BG = "#3D57A1"
KPI_BORDER = "#E6E8ED"

base_font = {"fontFamily": "Arial, sans-serif"}


df = pd.read_csv(CSV_PATH)

if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
if "country" in df.columns:
    df["country"] = df["country"].astype(str)

possible_cases_cols = ["cases", "incident_cases", "incidence_cases", "incidence_count", "incident_count"]
possible_deaths_cols = ["deaths", "death_count", "mortality_count", "mortality_deaths"]

cases_col = next((c for c in possible_cases_cols if c in df.columns), None)
deaths_col = next((c for c in possible_deaths_cols if c in df.columns), None)

if cases_col is None and "incidence_asr" in df.columns and "population" in df.columns:
    df["cases_est"] = df["incidence_asr"] * df["population"] / 100000.0
    cases_col = "cases_est"

if deaths_col is None and "mortality_asr" in df.columns and "population" in df.columns:
    df["deaths_est"] = df["mortality_asr"] * df["population"] / 100000.0
    deaths_col = "deaths_est"

if cases_col is not None and "population" in df.columns:
    df["incidence_per_100k"] = (df[cases_col] / df["population"]) * 100000.0
else:
    df["incidence_per_100k"] = np.nan

if deaths_col is not None and "population" in df.columns:
    df["mortality_per_100k"] = (df[deaths_col] / df["population"]) * 100000.0
else:
    df["mortality_per_100k"] = np.nan

if cases_col is not None and deaths_col is not None:
    df["cfr"] = df[deaths_col] / df[cases_col]
    df["mir"] = df[deaths_col] / df[cases_col]
else:
    df["cfr"] = np.nan
    df["mir"] = np.nan

df = df.sort_values(["country", "year"])

for col in ["incidence_per_100k", "mortality_per_100k"]:
    df[f"{col}_yoy_pct"] = df.groupby("country")[col].pct_change() * 100.0

if cases_col is not None:
    df["cases_yoy_pct"] = df.groupby("country")[cases_col].pct_change() * 100.0
else:
    df["cases_yoy_pct"] = np.nan

if deaths_col is not None:
    df["deaths_yoy_pct"] = df.groupby("country")[deaths_col].pct_change() * 100.0
else:
    df["deaths_yoy_pct"] = np.nan


years = sorted(df["year"].dropna().unique().tolist())
countries = sorted(df["country"].dropna().unique().tolist())

default_year = int(years[-1]) if years else 2023
default_country = "France" if "France" in countries else (countries[0] if countries else None)

metric_options = [
    {"label": "Cas", "value": "cases"},
    {"label": "Décès", "value": "deaths"},
    {"label": "Incidence par 100 000", "value": "incidence_per_100k"},
    {"label": "Mortalité par 100 000", "value": "mortality_per_100k"},
]
if "incidence_asr" in df.columns:
    metric_options.append({"label": "Incidence ASR", "value": "incidence_asr"})
if "mortality_asr" in df.columns:
    metric_options.append({"label": "Mortalité ASR", "value": "mortality_asr"})


def fmt_int(x):
    if pd.isna(x):
        return "NA"
    return f"{int(round(float(x))):,}".replace(",", " ")


def fmt_float(x, nd=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def fmt_pct(x, nd=1):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f} %"


def get_metric_series_col(metric_value: str):
    if metric_value == "cases":
        return cases_col
    if metric_value == "deaths":
        return deaths_col
    return metric_value


def empty_fig(message: str):
    fig = px.line()
    fig.update_layout(
        title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
    )
    return fig


def apply_axis_format(fig, y_title="Valeur"):
    fig.update_layout(
        title=None,
        margin=dict(l=18, r=10, t=10, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Année",
            showline=True,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            tickmode="linear",
            dtick=1,
            tickangle=-35,
        ),
        yaxis=dict(
            title=y_title,
            showline=True,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            nticks=6,
        ),
        legend_title_text="",
    )
    return fig


WHO_DEFS = {
    "Incidence": (
        "Nombre d’instances (taux d’occurrence) d’une maladie qui commence, ou de personnes qui tombent malades, "
        "pendant une période donnée dans une population spécifiée."
    ),
    "Mortalité": (
        "Taux de décès dans une population, calculé comme le nombre de décès × 1000, "
        "divisé par la population au milieu de l’intervalle."
    ),
    "Ratio de létalité": "Proportion de personnes diagnostiquées avec une maladie qui en meurent.",
    "Incidence ASR": "Taux standardisé sur l’âge : taux qu’aurait une population si elle avait une structure d’âge standard.",
    "Mortalité ASR": (
        "Taux de mortalité standardisé sur l’âge : moyenne pondérée des taux de mortalité spécifiques par âge "
        "(par 100 000), les poids étant les proportions par groupes d’âge d’une population standard."
    ),
    "cfr": "Case fatality ratio (CFR) : proportion de personnes diagnostiquées avec une maladie qui en meurent.",
    "yoy_pct": (
        "Variation en pourcentage par rapport à l’année précédente : "
        "((valeur année N − valeur année N−1) / valeur année N−1) × 100."
    ),
}


def safe_weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = s.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float((s[mask] * w[mask]).sum() / w[mask].sum())


def world_series_by_year() -> pd.DataFrame:
    cols_needed = ["year"]
    if "population" in df.columns:
        cols_needed.append("population")
    if cases_col and cases_col in df.columns:
        cols_needed.append(cases_col)
    if deaths_col and deaths_col in df.columns:
        cols_needed.append(deaths_col)

    d = df[cols_needed].copy().dropna(subset=["year"])

    g = d.groupby("year", as_index=False).agg(
        population=("population", "sum") if "population" in d.columns else ("year", "size"),
        cases=(cases_col, "sum") if (cases_col and cases_col in d.columns) else ("year", "size"),
        deaths=(deaths_col, "sum") if (deaths_col and deaths_col in d.columns) else ("year", "size"),
    )

    if not ("population" in df.columns):
        g["population"] = np.nan
    if not (cases_col and cases_col in df.columns):
        g["cases"] = np.nan
    if not (deaths_col and deaths_col in df.columns):
        g["deaths"] = np.nan

    g["incidence_per_100k"] = np.where(
        (g["population"].notna()) & (g["population"] != 0) & (g["cases"].notna()),
        (g["cases"] / g["population"]) * 100000.0,
        np.nan,
    )
    g["mortality_per_100k"] = np.where(
        (g["population"].notna()) & (g["population"] != 0) & (g["deaths"].notna()),
        (g["deaths"] / g["population"]) * 100000.0,
        np.nan,
    )

    g["cfr"] = np.where(
        (g["cases"].notna()) & (g["cases"] != 0) & (g["deaths"].notna()),
        g["deaths"] / g["cases"],
        np.nan,
    )

    g = g.sort_values("year")
    g["cases_yoy_pct"] = g["cases"].pct_change() * 100.0
    g["deaths_yoy_pct"] = g["deaths"].pct_change() * 100.0
    g["incidence_per_100k_yoy_pct"] = g["incidence_per_100k"].pct_change() * 100.0
    g["mortality_per_100k_yoy_pct"] = g["mortality_per_100k"].pct_change() * 100.0

    return g


WORLD_TS = world_series_by_year()


def world_row_for_year(year: int) -> dict:
    row = WORLD_TS[WORLD_TS["year"] == year]
    if row.empty:
        return {
            k: np.nan
            for k in [
                "cases",
                "deaths",
                "population",
                "incidence_per_100k",
                "mortality_per_100k",
                "cfr",
                "cases_yoy_pct",
                "deaths_yoy_pct",
                "incidence_per_100k_yoy_pct",
                "mortality_per_100k_yoy_pct",
                "incidence_asr",
                "mortality_asr",
                "urban_pop_pct",
                "gdp_per_capita",
                "health_exp_gdp",
            ]
        }

    out = row.iloc[0].to_dict()

    d = df[df["year"] == year].copy()
    if "population" in d.columns:
        pop = d["population"]
        out["incidence_asr"] = safe_weighted_mean(d["incidence_asr"], pop) if "incidence_asr" in d.columns else np.nan
        out["mortality_asr"] = safe_weighted_mean(d["mortality_asr"], pop) if "mortality_asr" in d.columns else np.nan
        out["urban_pop_pct"] = safe_weighted_mean(d["urban_pop_pct"], pop) if "urban_pop_pct" in d.columns else np.nan
        out["gdp_per_capita"] = safe_weighted_mean(d["gdp_per_capita"], pop) if "gdp_per_capita" in d.columns else np.nan
        out["health_exp_gdp"] = safe_weighted_mean(d["health_exp_gdp"], pop) if "health_exp_gdp" in d.columns else np.nan
    else:
        out["incidence_asr"] = np.nan
        out["mortality_asr"] = np.nan
        out["urban_pop_pct"] = np.nan
        out["gdp_per_capita"] = np.nan
        out["health_exp_gdp"] = np.nan

    return out


app = Dash(__name__, external_stylesheets=[FA_CSS])
server = app.server

app.title = "TABLEAU DE BORD ÉPIDÉMIOLOGIQUE MONDIAL DU CANCER DE SEIN : 2010 - 2023"

center_title_h4 = {"marginTop": "0", "color": TEXT_DARK, "textAlign": "center", "fontWeight": "900"}

main_title_card_style = {
    "backgroundColor": TITLE_CARD_BG,
    "borderRadius": "20px",
    "padding": "18px 18px",
    "boxShadow": "0 10px 25px rgba(0,0,0,0.20)",
    "marginBottom": "12px",
}
main_title_text_style = {
    "textAlign": "center",
    "fontWeight": "900",
    "fontSize": "30px",
    "color": TEXT_DARK,
    "letterSpacing": "0.8px",
    "textShadow": "0 3px 0 rgba(255,255,255,0.75), 0 10px 18px rgba(0,0,0,0.28)",
    "margin": "0",
    "lineHeight": "1.15",
}

kpi_card_style = {
    "backgroundColor": KPI_BG,
    "border": f"1px solid {KPI_BORDER}",
    "borderRadius": "6px",
    "padding": "12px 10px",
    "boxShadow": "none",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "space-between",
    "minHeight": "120px",
}

kpi_title_style = {
    "fontWeight": "800",
    "fontSize": "14px",
    "color": "#FFFFFF",
    "textAlign": "center",
    "marginBottom": "10px",
    "lineHeight": "1.4",
}
kpi_value_style = {
    "fontWeight": "900",
    "fontSize": "34px",
    "color": "#FFFFFF",
    "textAlign": "center",
    "lineHeight": "1.3",
    "marginBottom": "12px",
}
kpi_footer_style = {
    "fontWeight": "700",
    "fontSize": "11px",
    "color": "#FFFFFF",
    "opacity": "0.95",
    "textAlign": "center",
    "lineHeight": "1.5",
    "marginBottom": "4px",
}
kpi_subline_style = {
    "fontWeight": "700",
    "fontSize": "12px",
    "color": "#FFFFFF",
    "opacity": "0.95",
    "textAlign": "center",
    "marginTop": "6px",
    "lineHeight": "1.5",
}


app.layout = html.Div(
    style={**base_font, "display": "flex", "height": "100vh"},
    children=[
        html.Div(
            style={
                "width": "290px",
                "backgroundColor": SIDEBAR_BG,
                "color": TEXT_LIGHT,
                "padding": "18px 16px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "14px",
                "height": "100vh",
                "boxSizing": "border-box",
                "borderRadius": "18px",
                "margin": "12px",
            },
            children=[
                html.Div(
                    style={"textAlign": "center", "marginBottom": "4px"},
                    children=[
                        html.I(
                            className="fa-solid fa-ribbon",
                            style={"fontSize": "120px", "color": PINK_SLOGAN, "display": "block", "margin": "8px auto 0 auto"},
                        ),
                        html.Div(
                            "Des données pour sauver des vies",
                            style={"marginTop": "10px", "fontWeight": "900", "fontSize": "13px", "color": PINK_SLOGAN},
                        ),
                    ],
                ),
                html.Hr(style={"borderColor": "rgba(255,255,255,0.25)"}),

                html.Div("Niveau d’analyse", style={"fontWeight": "900", "fontSize": "14px"}),
                dcc.RadioItems(
                    id="scope_rb",
                    options=[{"label": "Pays", "value": "country"}, {"label": "Monde", "value": "world"}],
                    value="country",
                    labelStyle={"display": "block", "marginBottom": "6px"},
                    style={"color": TEXT_LIGHT, "fontWeight": "700"},
                ),

                html.Div("Pays", style={"fontWeight": "900", "fontSize": "14px"}),
                dcc.Dropdown(
                    id="country_dd",
                    options=[{"label": c, "value": c} for c in countries],
                    value=default_country,
                    clearable=False,
                    searchable=True,
                    style={"color": TEXT_DARK},
                ),

                html.Div("Année", style={"fontWeight": "900", "marginTop": "8px", "fontSize": "14px"}),
                dcc.Dropdown(
                    id="year_dd",
                    options=[{"label": str(y), "value": int(y)} for y in years],
                    value=default_year,
                    clearable=False,
                    style={"color": TEXT_DARK},
                ),

                html.Div("Métrique principale", style={"fontWeight": "900", "marginTop": "8px", "fontSize": "14px"}),
                dcc.Dropdown(
                    id="metric_dd",
                    options=metric_options,
                    value="incidence_per_100k",
                    clearable=False,
                    style={"color": TEXT_DARK},
                ),

                html.Div(
                    style={"marginTop": "auto"},
                    children=[
                        html.Hr(style={"borderColor": "rgba(255,255,255,0.25)", "margin": "12px 0"}),
                        html.Div("Membres du groupe", style={"fontWeight": "900", "marginBottom": "6px", "fontSize": "13px"}),
                        html.Ul(
                            style={"margin": "0", "paddingLeft": "18px", "opacity": "0.95", "lineHeight": "1.55", "fontSize": "12.5px"},
                            children=[
                                html.Li("Joseph Giovanni AGBAHOUNGBA"),
                                html.Li("Hippolyte ADECHIAN"),
                                html.Li("Elvira Francheska KENGNI"),
                                html.Li("Qais KAZIMI"),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={
                "flex": "1",
                "backgroundColor": CONTENT_BG,
                "padding": "18px",
                "height": "100vh",
                "overflowY": "auto",
                "boxSizing": "border-box",
                "borderRadius": "18px",
                "margin": "12px 12px 12px 0",
            },
            children=[
                html.Div(
                    style=main_title_card_style,
                    children=[
                        html.Div(
                            "TABLEAU DE BORD ÉPIDÉMIOLOGIQUE MONDIAL DU CANCER DE SEIN : 2010 - 2023",
                            style=main_title_text_style,
                        ),
                        html.Div(
                            "Source de données : Institute for Health Metrics and Evaluation (IHME) - Global Burden of Disease (GBD).",
                            style={"textAlign": "center", "marginTop": "8px", "fontSize": "13px", "color": "#000000", "fontWeight": "600", "opacity": "0.9"},
                        ),
                    ],
                ),

                html.Div(
                    id="subtitle",
                    style={"color": TEXT_DARK, "opacity": "0.95", "marginBottom": "14px", "fontWeight": "900", "textAlign": "center", "fontSize": "18px"},
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(210px, 1fr))", "gap": "10px", "marginBottom": "14px"},
                    children=[
                        html.Div(id="kpi_cases", style=kpi_card_style),
                        html.Div(id="kpi_deaths", style=kpi_card_style),
                        html.Div(id="kpi_rates", style=kpi_card_style),
                        html.Div(id="kpi_ratio", style=kpi_card_style),
                    ],
                ),

                html.Div(
                    style={"backgroundColor": CARD_TREND, "borderRadius": "16px", "padding": "14px", "boxShadow": "0 6px 18px rgba(0,0,0,0.12)", "marginBottom": "14px"},
                    children=[
                        html.Div(id="trend_block_title", style={**center_title_h4, "marginBottom": "10px"}),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                            children=[
                                html.Div(
                                    style={"backgroundColor": "#FFFFFF", "borderRadius": "14px", "padding": "10px", "boxShadow": "0 4px 14px rgba(0,0,0,0.10)"},
                                    children=[
                                        html.H4(id="cases_trend_title", style={**center_title_h4, "fontSize": "16px"}),
                                        dcc.Graph(id="cases_trend_fig", config={"displayModeBar": False}),
                                    ],
                                ),
                                html.Div(
                                    style={"backgroundColor": "#FFFFFF", "borderRadius": "14px", "padding": "10px", "boxShadow": "0 4px 14px rgba(0,0,0,0.10)"},
                                    children=[
                                        html.H4(id="deaths_trend_title", style={**center_title_h4, "fontSize": "16px"}),
                                        dcc.Graph(id="deaths_trend_fig", config={"displayModeBar": False}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1.2fr 0.8fr", "gap": "14px"},
                    children=[
                        html.Div(
                            style={"backgroundColor": CARD_MAP, "borderRadius": "16px", "padding": "14px", "boxShadow": "0 6px 18px rgba(0,0,0,0.12)"},
                            children=[
                                html.H4(id="map_title", style=center_title_h4),
                                dcc.Graph(id="map_fig", config={"displayModeBar": False}),
                            ],
                        ),
                        html.Div(
                            style={"backgroundColor": "#FFFFFF", "borderRadius": "16px", "padding": "14px", "boxShadow": "0 6px 18px rgba(0,0,0,0.12)"},
                            children=[
                                html.H4(id="top10_title", style=center_title_h4),
                                dcc.Graph(id="top10_fig", config={"displayModeBar": False}),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    style={"marginTop": "14px", "backgroundColor": "#FFFFFF", "borderRadius": "16px", "padding": "14px", "boxShadow": "0 6px 18px rgba(0,0,0,0.12)"},
                    children=[
                        html.H4(id="epi_title", style=center_title_h4),
                        html.Div(id="epi_table_container"),
                    ],
                ),

                html.Div(
                    style={"marginTop": "14px", "backgroundColor": "#FFFFFF", "borderRadius": "16px", "padding": "14px", "boxShadow": "0 6px 18px rgba(0,0,0,0.12)"},
                    children=[
                        html.H4("Notions clés (OMS)", style=center_title_h4),
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            notion,
                                            style={"cursor": "pointer", "fontWeight": "900", "color": TEXT_DARK, "fontSize": "14px", "padding": "8px 6px"},
                                        ),
                                        html.Div(
                                            WHO_DEFS[notion],
                                            style={"padding": "10px 10px 12px 10px", "color": TEXT_DARK, "fontSize": "13px", "lineHeight": "1.5"},
                                        ),
                                    ],
                                    style={"border": "1px solid rgba(0,0,0,0.10)", "borderRadius": "12px", "marginBottom": "10px", "backgroundColor": "#F8FBFF", "padding": "6px 8px"},
                                )
                                for notion in WHO_DEFS.keys()
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("country_dd", "disabled"),
    Input("scope_rb", "value"),
)
def toggle_country_dd(scope_value):
    return scope_value == "world"


@app.callback(
    Output("subtitle", "children"),
    Output("kpi_cases", "children"),
    Output("kpi_deaths", "children"),
    Output("kpi_rates", "children"),
    Output("kpi_ratio", "children"),
    Output("trend_block_title", "children"),
    Output("cases_trend_title", "children"),
    Output("deaths_trend_title", "children"),
    Output("map_title", "children"),
    Output("top10_title", "children"),
    Output("epi_title", "children"),
    Output("cases_trend_fig", "figure"),
    Output("deaths_trend_fig", "figure"),
    Output("map_fig", "figure"),
    Output("top10_fig", "figure"),
    Output("epi_table_container", "children"),
    Input("scope_rb", "value"),
    Input("country_dd", "value"),
    Input("year_dd", "value"),
    Input("metric_dd", "value"),
)
def update(scope, country, year, metric_choice):
    if year is None:
        return (
            "Sélectionnez une année",
            html.Div("NA"),
            html.Div("NA"),
            html.Div("NA"),
            html.Div("NA"),
            "Tendance annuelle",
            "Cas",
            "Décès",
            "Carte mondiale",
            "Top 10 pays",
            "Fiche épidémiologique",
            empty_fig("Sélection requise"),
            empty_fig("Sélection requise"),
            empty_fig("Sélection requise"),
            empty_fig("Sélection requise"),
            html.Div("Sélection requise"),
        )

    year = int(year)
    label_entity = "Monde" if scope == "world" else (country if country else "Pays")
    subtitle = f"{label_entity} : {year}"

    trend_block_title = f"Tendance annuelle ({label_entity})"
    cases_trend_title = f"Cas (année par année) – référence {year}"
    deaths_trend_title = f"Décès (année par année) – référence {year}"
    map_title = f"Carte mondiale pour l’année {year}"
    top10_title = f"Top 10 pays pour l’année {year}"
    epi_title = f"Fiche épidémiologique ({label_entity}) - année {year}"

    def kpi_box(title, big_value, footer_lines):
        footer_children = []
        for i, t in enumerate(footer_lines):
            footer_children.append(
                html.Div(
                    t,
                    style=kpi_footer_style if i == 0 else {**kpi_footer_style, "marginTop": "6px"},
                )
            )
        return html.Div(
            [
                html.Div(title, style=kpi_title_style),
                html.Div(big_value, style=kpi_value_style),
                html.Div(footer_children, style={"marginTop": "2px"}),
            ],
            style={"height": "100%"},
        )

    if scope == "world":
        w = world_row_for_year(year)

        cases_v = w.get("cases", np.nan)
        deaths_v = w.get("deaths", np.nan)

        inc100_v = w.get("incidence_per_100k", np.nan)
        mort100_v = w.get("mortality_per_100k", np.nan)

        inc_asr_v = w.get("incidence_asr", np.nan)
        mort_asr_v = w.get("mortality_asr", np.nan)

        cfr_v = w.get("cfr", np.nan)

        cases_yoy = w.get("cases_yoy_pct", np.nan)
        deaths_yoy = w.get("deaths_yoy_pct", np.nan)
        inc_yoy = w.get("incidence_per_100k_yoy_pct", np.nan)
        mort_yoy = w.get("mortality_per_100k_yoy_pct", np.nan)

        ts = WORLD_TS.copy()

        cases_label = "Cas"
        deaths_label = "Décès"

        kpi_cases = kpi_box("Cas estimés", fmt_int(cases_v), [f"Variation annuelle : {fmt_pct(cases_yoy)}"])
        kpi_deaths = kpi_box("Décès estimés", fmt_int(deaths_v), [f"Variation annuelle : {fmt_pct(deaths_yoy)}"])

        kpi_rates = kpi_box(
            "Taux bruts par 100 000",
            fmt_float(inc100_v),
            [
                f"Incidence - variation : {fmt_pct(inc_yoy)}",
                f"Mortalité : {fmt_float(mort100_v)} (var. {fmt_pct(mort_yoy)})",
            ],
        )

        kpi_ratio = kpi_box(
            "Indicateurs",
            fmt_float(cfr_v, 3),
            [
                "Ratio de létalité (CFR)",
                f"Incidence ASR : {fmt_float(inc_asr_v)}",
                f"Mortalité ASR : {fmt_float(mort_asr_v)}",
            ],
        )

    else:
        if country is None:
            return (
                "Sélectionnez un pays et une année",
                html.Div("NA"),
                html.Div("NA"),
                html.Div("NA"),
                html.Div("NA"),
                "Tendance annuelle",
                "Cas",
                "Décès",
                "Carte mondiale",
                "Top 10 pays",
                "Fiche épidémiologique",
                empty_fig("Sélection requise"),
                empty_fig("Sélection requise"),
                empty_fig("Sélection requise"),
                empty_fig("Sélection requise"),
                html.Div("Sélection requise"),
            )

        dff_country = df[df["country"] == country].copy()
        row = dff_country[dff_country["year"] == year]

        def get_val(col):
            if col is None or col not in df.columns or row.empty:
                return np.nan
            return row[col].iloc[0]

        cases_v = get_val(cases_col)
        deaths_v = get_val(deaths_col)

        inc100_v = get_val("incidence_per_100k")
        mort100_v = get_val("mortality_per_100k")

        inc_asr_v = get_val("incidence_asr") if "incidence_asr" in df.columns else np.nan
        mort_asr_v = get_val("mortality_asr") if "mortality_asr" in df.columns else np.nan

        cfr_v = get_val("cfr")

        cases_yoy = get_val("cases_yoy_pct")
        deaths_yoy = get_val("deaths_yoy_pct")
        inc_yoy = get_val("incidence_per_100k_yoy_pct")
        mort_yoy = get_val("mortality_per_100k_yoy_pct")

        cases_label = "Cas" if not (cases_col or "").endswith("_est") else "Cas estimés"
        deaths_label = "Décès" if not (deaths_col or "").endswith("_est") else "Décès estimés"

        ts = dff_country.sort_values("year").copy()

        kpi_cases = kpi_box(cases_label, fmt_int(cases_v), [f"Variation annuelle : {fmt_pct(cases_yoy)}"])
        kpi_deaths = kpi_box(deaths_label, fmt_int(deaths_v), [f"Variation annuelle : {fmt_pct(deaths_yoy)}"])

        kpi_rates = kpi_box(
            "Taux bruts par 100 000",
            fmt_float(inc100_v),
            [
                f"Incidence - variation : {fmt_pct(inc_yoy)}",
                f"Mortalité : {fmt_float(mort100_v)} (var. {fmt_pct(mort_yoy)})",
            ],
        )

        kpi_ratio = kpi_box(
            "Indicateurs",
            fmt_float(cfr_v, 3),
            [
                "Ratio de létalité (CFR)",
                f"Incidence ASR : {fmt_float(inc_asr_v)}",
                f"Mortalité ASR : {fmt_float(mort_asr_v)}",
            ],
        )

    # =========================
    # COURBES DE TENDANCE (SANS COMPARAISON MONDE)
    # =========================

    if scope == "world":
        if ts[["year", "cases"]].dropna().empty:
            cases_trend_fig = empty_fig("Cas indisponibles")
        else:
            dplot = ts[["year", "cases"]].dropna()
            cases_trend_fig = px.line(dplot, x="year", y="cases", markers=True)
            cases_trend_fig.update_traces(line=dict(color=CASES_COLOR), marker=dict(color=CASES_COLOR))
            cases_trend_fig = apply_axis_format(cases_trend_fig, y_title="Cas")

        if ts[["year", "deaths"]].dropna().empty:
            deaths_trend_fig = empty_fig("Décès indisponibles")
        else:
            dplot = ts[["year", "deaths"]].dropna()
            deaths_trend_fig = px.line(dplot, x="year", y="deaths", markers=True)
            deaths_trend_fig.update_traces(line=dict(color=DEATHS_COLOR), marker=dict(color=DEATHS_COLOR))
            deaths_trend_fig = apply_axis_format(deaths_trend_fig, y_title="Décès")

    else:
        # --- CAS : Pays uniquement ---
        if cases_col is None or cases_col not in df.columns:
            cases_trend_fig = empty_fig("Cas indisponibles")
        else:
            dplot = ts[["year", cases_col]].dropna()
            if dplot.empty:
                cases_trend_fig = empty_fig("Aucune donnée")
            else:
                cases_trend_fig = px.line(dplot, x="year", y=cases_col, markers=True)
                cases_trend_fig.update_traces(line=dict(color=CASES_COLOR), marker=dict(color=CASES_COLOR))
                cases_trend_fig = apply_axis_format(cases_trend_fig, y_title=cases_label)

        # --- DÉCÈS : Pays uniquement ---
        if deaths_col is None or deaths_col not in df.columns:
            deaths_trend_fig = empty_fig("Décès indisponibles")
        else:
            dplot = ts[["year", deaths_col]].dropna()
            if dplot.empty:
                deaths_trend_fig = empty_fig("Aucune donnée")
            else:
                deaths_trend_fig = px.line(dplot, x="year", y=deaths_col, markers=True)
                deaths_trend_fig.update_traces(line=dict(color=DEATHS_COLOR), marker=dict(color=DEATHS_COLOR))
                deaths_trend_fig = apply_axis_format(deaths_trend_fig, y_title=deaths_label)

    # =========================
    # CARTE + TOP 10
    # =========================
    metric_col = get_metric_series_col(metric_choice)
    dff_year = df[df["year"] == year].copy()

    if metric_col is None or metric_col not in df.columns or "iso3" not in df.columns:
        map_fig = empty_fig("Carte indisponible")
    else:
        map_df = dff_year[["iso3", "country", metric_col]].dropna()
        map_fig = px.choropleth(map_df, locations="iso3", color=metric_col, hover_name="country")

        if scope == "country":
            row_sel = dff_year[dff_year["country"] == country]
            iso3_sel = None
            if not row_sel.empty and "iso3" in row_sel.columns and pd.notna(row_sel["iso3"].iloc[0]):
                iso3_sel = row_sel["iso3"].iloc[0]

            if isinstance(iso3_sel, str) and len(iso3_sel) == 3:
                sc = px.scatter_geo(
                    pd.DataFrame({"iso3": [iso3_sel], "country": [country]}),
                    locations="iso3",
                    locationmode="ISO-3",
                    text="country",
                )
                sc.update_traces(
                    marker=dict(size=MAP_MARK_SIZE, color=MAP_MARK_YELLOW, symbol="circle"),
                    textposition="top center",
                    textfont=dict(color=MAP_MARK_YELLOW, size=13, family="Arial Black"),
                    hovertemplate="%{text}<extra></extra>",
                )
                map_fig.add_trace(sc.data[0])

        map_fig.update_layout(
            title=None,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(title=""),
            geo=dict(showframe=False, showcoastlines=False),
        )

    if metric_col is None or metric_col not in df.columns:
        top10_fig = empty_fig("Top 10 indisponible")
    else:
        top_df = dff_year[["country", metric_col]].dropna().sort_values(metric_col, ascending=False).head(10)
        if top_df.empty:
            top10_fig = empty_fig("Aucune donnée")
        else:
            top10_fig = px.bar(top_df, x=metric_col, y="country", orientation="h")
            top10_fig.update_layout(
                title=None,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Valeur",
                yaxis_title="",
            )

    # =========================
    # TABLE ÉPIDÉMIO
    # =========================
    if scope == "world":
        w = world_row_for_year(year)
        table_data = {
            "entity": "Monde",
            "year": year,
            "cases": w.get("cases", np.nan),
            "deaths": w.get("deaths", np.nan),
            "population": w.get("population", np.nan),
            "urban_pop_pct": w.get("urban_pop_pct", np.nan),
            "gdp_per_capita": w.get("gdp_per_capita", np.nan),
            "health_exp_gdp": w.get("health_exp_gdp", np.nan),
            "incidence_per_100k": w.get("incidence_per_100k", np.nan),
            "mortality_per_100k": w.get("mortality_per_100k", np.nan),
            "incidence_asr": w.get("incidence_asr", np.nan),
            "mortality_asr": w.get("mortality_asr", np.nan),
            "cfr": w.get("cfr", np.nan),
            "cases_yoy_pct": w.get("cases_yoy_pct", np.nan),
            "deaths_yoy_pct": w.get("deaths_yoy_pct", np.nan),
            "incidence_per_100k_yoy_pct": w.get("incidence_per_100k_yoy_pct", np.nan),
            "mortality_per_100k_yoy_pct": w.get("mortality_per_100k_yoy_pct", np.nan),
        }
        row_disp = pd.DataFrame([table_data])

    else:
        row_disp = dff_year[dff_year["country"] == country].copy()
        if row_disp.empty:
            epi_table = html.Div("Aucune donnée pour cette combinaison pays et année.")
            return (
                subtitle,
                kpi_cases,
                kpi_deaths,
                kpi_rates,
                kpi_ratio,
                trend_block_title,
                cases_trend_title,
                deaths_trend_title,
                map_title,
                top10_title,
                epi_title,
                cases_trend_fig,
                deaths_trend_fig,
                map_fig,
                top10_fig,
                epi_table,
            )

        table_cols = []

        def add_col(c):
            if c in df.columns:
                table_cols.append(c)

        add_col("country")
        add_col("iso3")
        add_col("year")
        if cases_col:
            add_col(cases_col)
        if deaths_col:
            add_col(deaths_col)
        add_col("population")
        add_col("urban_pop_pct")
        add_col("gdp_per_capita")
        add_col("health_exp_gdp")
        add_col("incidence_per_100k")
        add_col("mortality_per_100k")
        add_col("incidence_asr")
        add_col("mortality_asr")
        add_col("cfr")
        add_col("cases_yoy_pct")
        add_col("deaths_yoy_pct")
        add_col("incidence_per_100k_yoy_pct")
        add_col("mortality_per_100k_yoy_pct")

        row_disp = row_disp[table_cols].copy()

        rename_map = {}
        if cases_col == "cases_est":
            rename_map["cases_est"] = "cas estimés"
        if deaths_col == "deaths_est":
            rename_map["deaths_est"] = "décès estimés"
        row_disp = row_disp.rename(columns=rename_map)

    epi_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in row_disp.columns],
        data=row_disp.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Arial", "fontSize": "13px", "padding": "8px", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "700", "backgroundColor": "#F2F6FF"},
    )

    return (
        subtitle,
        kpi_cases,
        kpi_deaths,
        kpi_rates,
        kpi_ratio,
        trend_block_title,
        cases_trend_title,
        deaths_trend_title,
        map_title,
        top10_title,
        epi_title,
        cases_trend_fig,
        deaths_trend_fig,
        map_fig,
        top10_fig,
        epi_table,
    )


if __name__ == "__main__":
    app.run(debug=True)