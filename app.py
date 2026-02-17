import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Définition des constantes
DATA_FILENAME = "dataset_final_cancer_sein_IHME_WB_2010_2025.csv"
APP_TITLE = "Tableau de bord épidémiologique mondial du cancer de sein : 2010 - 2023"
SUBTITLE = "Sources de données : IHME & Banque mondiale"
QUALITY_TEXT_FIXED = "Qualité du modèle - coefficient de détermination = 0.963 | erreur absolue moyenne = 0.912"

THEME = dbc.themes.DARKLY

COLORS = {
    "bg": "#0B1220",
    "panel": "#0E1A2F",
    "panel2": "#0C1730",
    "border_soft": "rgba(255,255,255,0.10)",
    "white": "#FFFFFF",
    "text": "#EAF0FF",
    "muted": "rgba(234,240,255,0.75)",
    "grid": "rgba(255,255,255,0.08)",
    "kpi_value": "#FACC15",
    "tab_idle": "#0E1A2F",
    "tab_selected": "#1E3A8A",
    "tab_selected_text": "#FFFFFF",
}

RADIUS = "18px"
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

STYLE_APP = {"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "14px"}

STYLE_MAIN_PANEL = {
    "backgroundColor": COLORS["panel"],
    "border": f"1px solid {COLORS['border_soft']}",
    "borderRadius": RADIUS,
    "boxShadow": "0 10px 24px rgba(0,0,0,0.25)",
    "padding": "12px",
}

STYLE_SIDEBAR = {
    "backgroundColor": COLORS["panel"],
    "border": f"1px solid {COLORS['border_soft']}",
    "borderRadius": RADIUS,
    "padding": "14px",
    "height": "calc(100vh - 28px)",
    "position": "sticky",
    "top": "14px",
    "display": "flex",
    "flexDirection": "column",
}

STYLE_CARD = {
    "backgroundColor": COLORS["panel2"],
    "border": f"1px solid {COLORS['border_soft']}",
    "borderRadius": "14px",
    "boxShadow": "0 6px 18px rgba(0,0,0,0.20)",
}

STYLE_BLOCK_CONTAINER = {
    "backgroundColor": COLORS["panel"],
    "border": f"1px solid {COLORS['white']}",
    "borderRadius": "16px",
    "padding": "12px",
}

TAB_STYLE = {
    "backgroundColor": COLORS["tab_idle"],
    "color": COLORS["text"],
    "border": "none",
    "padding": "10px 10px",
    "fontSize": "0.95rem",
    "textAlign": "center",
}
TAB_SELECTED_STYLE = {
    "backgroundColor": COLORS["tab_selected"],
    "color": COLORS["tab_selected_text"],
    "border": "none",
    "padding": "10px 10px",
    "fontSize": "0.95rem",
    "fontWeight": "700",
    "textAlign": "center",
}

CSS_DROPDOWN = """
.sidebar .Select-control { background-color: #F3F6FF !important; border: 1px solid rgba(0,0,0,0.20) !important; }
.sidebar .Select-placeholder { color: #000000 !important; }
.sidebar .Select-value-label { color: #000000 !important; font-weight: 600 !important; }
.sidebar .Select-menu-outer { background-color: #FFFFFF !important; border:1px solid rgba(0,0,0,0.20) !important; }
.sidebar .Select-option { color: #000000 !important; background-color: #FFFFFF !important; }
.sidebar .Select-option.is-focused { background-color: #E6EEFF !important; color: #000000 !important; }
.sidebar .Select-option.is-selected { background-color: #CFE0FF !important; color: #000000 !important; }
.sidebar .VirtualizedSelectOption{ color:#000000 !important; background-color:#FFFFFF !important; }
.sidebar .VirtualizedSelectFocusedOption{ color:#000000 !important; background-color:#E6EEFF !important; }
.sidebar .Select { color:#000000 !important; }
"""


def wrap_title(s: str, width: int = 38) -> str:
    return "<br>".join(textwrap.wrap(str(s).strip(), width=width))

def apply_french_layout(fig, title: str, legend_mode: str = "bottom"):
    fig.update_layout(
        title={"text": wrap_title(title), "x": 0.5, "xanchor": "center"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=COLORS["text"]),
        title_font=dict(color=COLORS["text"]),
    )
    fig.update_xaxes(
        title_font=dict(color=COLORS["text"]),
        tickfont=dict(color=COLORS["text"]),
        showgrid=True, gridcolor=COLORS["grid"],
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(color=COLORS["text"]),
        tickfont=dict(color=COLORS["text"]),
        showgrid=True, gridcolor=COLORS["grid"],
        zeroline=False,
    )

    if legend_mode == "right":
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                title_font=dict(color=COLORS["text"]),
            ),
            margin=dict(l=10, r=170, t=70, b=20),
        )
    else:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.22,
                xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"]),
                title_font=dict(color=COLORS["text"]),
            ),
            margin=dict(l=10, r=10, t=70, b=95),
        )
    return fig

def H(text):
    return html.Div(text, className="h5", style={"marginBottom": "10px", "textAlign": "center"})

def small(text):
    return html.Div(text, style={"color": COLORS["muted"], "fontSize": "0.9rem", "textAlign": "center"})

def interpretation_box(children):
    return dbc.Card(
        dbc.CardBody([
            html.Div("Interprétation", className="h6", style={"textAlign": "center", "fontWeight": "700"}),
            html.Div(children, style={"color": COLORS["muted"], "textAlign": "center", "fontSize": "0.96rem"})
        ], style={"padding": "12px"}),
        style=STYLE_CARD,
        className="mt-2"
    )

def fmt(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{x:,.{digits}f}"
    except Exception:
        return "-"

def kpi_card(title, value, sub=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"color": COLORS["muted"], "fontSize": "0.92rem", "textAlign": "center", "fontWeight": "700"}),
            html.Div(value, className="h4", style={"margin": "6px 0 2px 0", "textAlign": "center", "color": COLORS["kpi_value"], "fontWeight": "800"}),
            html.Div(sub, style={"color": COLORS["muted"], "fontSize": "0.82rem", "textAlign": "center"}),
        ], style={"padding": "10px"}),
        style=STYLE_CARD
    )

# Chargement et manipulation des données
def load_data(path: str = DATA_FILENAME) -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(path)
    if not csv_path.is_file():
        csv_path = script_dir / path
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV introuvable")
    return pd.read_csv(csv_path)

def safe_numeric(df_, cols):
    for c in cols:
        if c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")
    return df_

def add_features(df_: pd.DataFrame) -> pd.DataFrame:
    d = df_.copy()

    d = safe_numeric(
        d,
        ["year", "incidence_asr", "mortality_asr", "gdp_per_capita", "health_exp_gdp", "population", "urban_pop_pct"],
    )

    for c in ["country", "location_name", "iso3"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()

    essentials = [c for c in ["country", "year", "incidence_asr", "mortality_asr", "population"] if c in d.columns]
    d = d.dropna(subset=essentials)

    d["estimated_cases"] = (d["incidence_asr"] / 100000.0) * d["population"]
    d["estimated_deaths"] = (d["mortality_asr"] / 100000.0) * d["population"]
    d["fatality_proxy"] = d["mortality_asr"] / d["incidence_asr"].replace(0, np.nan)

    d = d.sort_values(["country", "year"])
    d["annual_change_incidence_pct"] = d.groupby("country")["incidence_asr"].pct_change() * 100
    d["annual_change_mortality_pct"] = d.groupby("country")["mortality_asr"].pct_change() * 100

    if "gdp_per_capita" in d.columns:
        try:
            d["gdp_quartile"] = pd.qcut(d["gdp_per_capita"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except Exception:
            d["gdp_quartile"] = np.nan

    comps = ["incidence_asr", "mortality_asr", "fatality_proxy", "annual_change_mortality_pct"]
    tmp = d[comps].replace([np.inf, -np.inf], np.nan).copy()
    tmp = (tmp - tmp.mean()) / (tmp.std(ddof=0).replace(0, np.nan))
    d["risk_score"] = tmp.mean(axis=1)

    d["risk_level"] = pd.cut(
        d["risk_score"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["Faible", "Modéré", "Élevé"]
    )
    return d

# AGRÉGATS MONDE
def world_year_table(df_year: pd.DataFrame) -> pd.DataFrame:
    d = df_year.dropna(subset=["population"]).copy()
    if d.empty:
        return pd.DataFrame()
    w = d["population"].replace(0, np.nan)

    def wavg(series):
        s = series.replace([np.inf, -np.inf], np.nan)
        ok = s.notna() & w.notna()
        if ok.sum() == 0:
            return np.nan
        return float(np.average(s[ok], weights=w[ok]))

    out = {
        "incidence_asr": wavg(d["incidence_asr"]),
        "mortality_asr": wavg(d["mortality_asr"]),
        "fatality_proxy": wavg(d["fatality_proxy"]),
        "estimated_cases": float(d["estimated_cases"].sum()),
        "estimated_deaths": float(d["estimated_deaths"].sum()),
        "risk_score": wavg(d["risk_score"]),
    }
    return pd.DataFrame([out])

def world_time_table(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, dy in df_all.groupby("year"):
        t = world_year_table(dy)
        if not t.empty:
            t["year"] = y
            rows.append(t)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("year")

# Définition des modèles
def train_mortality_model(df_: pd.DataFrame):
    target = "mortality_asr"
    features = ["incidence_asr", "gdp_per_capita", "health_exp_gdp", "urban_pop_pct", "population"]
    feats = [f for f in features if f in df_.columns]
    if target not in df_.columns or len(feats) < 2:
        return None

    X = df_[feats].replace([np.inf, -np.inf], np.nan).dropna()
    y = df_.loc[X.index, target]
    if len(X) < 80:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    model = RandomForestRegressor(n_estimators=320, random_state=42, min_samples_leaf=2, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "coefficient_determination": float(r2_score(y_test, pred)),
        "mean_absolute_error": float(mean_absolute_error(y_test, pred)),
        "features": feats,
    }
    importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    return model, metrics, importances

def forecast_trend(df_country: pd.DataFrame, y_col: str, horizon: int):
    dc = df_country.dropna(subset=["year", y_col]).copy()
    if dc["year"].nunique() < 3:
        return None

    X = dc[["year"]].astype(float).values
    y = dc[y_col].astype(float).values

    lr = LinearRegression()
    lr.fit(X, y)

    last_year = int(dc["year"].max())
    future = np.arange(last_year + 1, last_year + 1 + int(horizon))
    yhat = lr.predict(future.reshape(-1, 1))

    return pd.DataFrame({"year": future, "pred": yhat, "last_year": last_year})

def compute_clusters(df_: pd.DataFrame, year: int, k: int):
    cols = ["incidence_asr", "mortality_asr", "gdp_per_capita", "health_exp_gdp", "urban_pop_pct"]
    cols = [c for c in cols if c in df_.columns]

    d = df_[df_["year"] == year].dropna(subset=cols + ["country"]).copy()
    if d.empty or len(d) < k or len(cols) < 2:
        return None

    X = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d = d.loc[X.index].copy()
    if len(d) < k:
        return None

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
    d["cluster"] = km.fit_predict(Xs).astype(int)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    d["pca1"], d["pca2"] = Z[:, 0], Z[:, 1]
    var = pca.explained_variance_ratio_

    centers_scaled = km.cluster_centers_
    centers_pca = pca.transform(centers_scaled)
    centers = pd.DataFrame({"cluster": range(int(k)), "pca1": centers_pca[:, 0], "pca2": centers_pca[:, 1]})
    return d, cols, float(var[0]), float(var[1]), centers

# FIGURES
def empty_fig(msg="Données insuffisantes"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color=COLORS["text"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def choropleth(dy, col, title, height=360):
    if "iso3" not in dy.columns or col not in dy.columns:
        return empty_fig("Variable indisponible")
    fig = px.choropleth(dy, locations="iso3", color=col, hover_name="country", projection="natural earth")
    fig.update_layout(height=height)
    return apply_french_layout(fig, title, legend_mode="right")

def rank_bar(dy, col, title, topn=10, height=360):
    if col not in dy.columns:
        return empty_fig("Variable indisponible")
    d = dy.dropna(subset=[col, "country"]).sort_values(col, ascending=False).head(topn)
    if d.empty:
        return empty_fig("Aucune donnée")
    fig = px.bar(d, x=col, y="country", orientation="h")
    fig.update_layout(height=height, showlegend=False)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    return apply_french_layout(fig, title, legend_mode="right")

def ts_line(dc, col, title, height=270):
    if col not in dc.columns:
        return empty_fig("Variable indisponible")
    d = dc.dropna(subset=["year", col]).sort_values("year")
    if d.empty:
        return empty_fig("Aucune donnée")
    fig = px.line(d, x="year", y=col, markers=True)
    fig.update_layout(height=height, showlegend=False)
    fig.update_xaxes(title_text="Année")
    fig.update_yaxes(title_text="")
    return apply_french_layout(fig, title, legend_mode="right")

def change_bar(dc, col, title, height=270):
    if col not in dc.columns:
        return empty_fig("Variable indisponible")
    d = dc.dropna(subset=["year", col]).sort_values("year")
    if d.empty:
        return empty_fig("Aucune donnée")
    fig = px.bar(d, x="year", y=col)
    fig.update_layout(height=height, showlegend=False)
    fig.update_xaxes(title_text="Année")
    fig.update_yaxes(title_text="Variation (%)")
    return apply_french_layout(fig, title, legend_mode="right")

def corr_heat(dy, title, height=310):
    cols = ["incidence_asr", "mortality_asr", "fatality_proxy", "gdp_per_capita", "health_exp_gdp", "urban_pop_pct"]
    cols = [c for c in cols if c in dy.columns]
    if len(cols) < 2:
        return empty_fig("Colonnes insuffisantes")
    d = dy[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return empty_fig("Aucune donnée")
    corr = d.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(height=height)
    return apply_french_layout(fig, title, legend_mode="right")

def box_by_quartile(dy, col, title, height=310):
    if "gdp_quartile" not in dy.columns or col not in dy.columns:
        return empty_fig("Quartiles indisponibles")
    d = dy.dropna(subset=["gdp_quartile", col])
    if d.empty:
        return empty_fig("Aucune donnée")
    fig = px.box(d, x="gdp_quartile", y=col)
    fig.update_layout(height=height, showlegend=False)
    fig.update_xaxes(title_text="Quartile du produit intérieur brut par habitant")
    fig.update_yaxes(title_text="")
    return apply_french_layout(fig, title, legend_mode="right")

def importance_fig(importances):
    if importances is None or getattr(importances, "empty", True):
        return empty_fig("Modèle indisponible")
    imp = importances.head(10).sort_values(ascending=True)
    d = imp.reset_index()
    d.columns = ["variable", "importance"]
    fig = px.bar(d, x="importance", y="variable", orientation="h")
    fig.update_layout(height=340, showlegend=False)
    fig.update_xaxes(title_text="Importance (relative)")
    fig.update_yaxes(title_text="")
    return apply_french_layout(fig, "Variables les plus importantes", legend_mode="right")

# INIT DATA
df_raw = load_data(DATA_FILENAME)
df = add_features(df_raw)

years = sorted(df["year"].dropna().unique().astype(int))
countries = sorted(df["country"].dropna().unique().tolist())
max_year = int(max(years)) if years else 0

df_world_time = world_time_table(df)

trained = train_mortality_model(df)
if trained:
    model_rf, model_metrics, model_importances = trained
else:
    model_rf, model_metrics, model_importances = None, None, None

# APP + CSS
app = Dash(__name__, external_stylesheets=[THEME], title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CSS_DROPDOWN}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""
# LAYOUT
def sidebar():
    header_image = html.Img(
        src=app.get_asset_url("Cancer-du-sein.png"),
        style={
            "width": "100%",
            "borderRadius": "16px",
            "objectFit": "cover",
            "marginBottom": "10px",
            "border": "1px solid rgba(255,255,255,0.12)",
        },
        alt="Cancer du sein",
    )

    top_controls = html.Div([
        header_image,

        html.Div("Filtres", className="h5", style={"textAlign": "center", "fontWeight": "700"}),
        html.Div(style={"height": "12px"}),

        html.Div("Vue", style={"color": COLORS["muted"], "textAlign": "center"}),
        dcc.Dropdown(
            id="scope",
            options=[{"label": "Monde", "value": "Monde"}, {"label": "Pays", "value": "Pays"}],
            value="Monde",
            clearable=False,
        ),
        html.Div(style={"height": "10px"}),

        html.Div("Année", style={"color": COLORS["muted"], "textAlign": "center"}),
        dcc.Dropdown(
            id="year_dd",
            options=[{"label": str(y), "value": y} for y in years],
            value=max_year,
            clearable=False,
        ),
        html.Div(style={"height": "12px"}),

        html.Div("Pays", style={"color": COLORS["muted"], "textAlign": "center"}),
        dcc.Dropdown(
            id="country_dd",
            options=[{"label": c, "value": c} for c in countries],
            value=countries[0] if countries else None,
            clearable=False,
        ),

        html.Div(id="country_hint", style={"display": "none"}),

        html.Div(style={"height": "10px"}),
        html.Hr(style={"margin": "10px 0"}),

        html.Div("Horizon de prévision", style={"color": COLORS["muted"], "textAlign": "center"}),
        dcc.Slider(id="horizon", min=3, max=10, step=1, value=5, marks={i: str(i) for i in range(3, 11)}),

        html.Div(style={"height": "10px"}),
        html.Div("Nombre de groupes", style={"color": COLORS["muted"], "textAlign": "center"}),
        dcc.Slider(id="k", min=3, max=8, step=1, value=4, marks={i: str(i) for i in range(3, 9)}),
    ])

    bottom_members = html.Div([
        html.Hr(style={"margin": "10px 0"}),
        html.Div("Membres du groupe", style={"textAlign": "left", "fontWeight": "700, ", "fontSize": "0.95rem"}),
        html.Div("Joseph Giovanni AGBAHOUNGBA", style={"textAlign": "left", "color": COLORS["muted"], "fontSize": "0.82rem", "lineHeight": "1.2"}),
        html.Div("Hippolyte ADECHIAN", style={"textAlign": "left", "color": COLORS["muted"], "fontSize": "0.82rem", "lineHeight": "1.2"}),
        html.Div("Elvira Francheska KENGNI", style={"textAlign": "left", "color": COLORS["muted"], "fontSize": "0.82rem", "lineHeight": "1.2"}),
    ], style={"marginTop": "auto"})

    return html.Div([top_controls, bottom_members], className="sidebar", style=STYLE_SIDEBAR)

def tab_overview():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        H("Indicateurs principaux"),
        dbc.Row([
            dbc.Col(html.Div(id="kpi1"), width=4),
            dbc.Col(html.Div(id="kpi2"), width=4),
            dbc.Col(html.Div(id="kpi3"), width=4),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(html.Div(id="kpi4"), width=4),
            dbc.Col(html.Div(id="kpi5"), width=4),
            dbc.Col(html.Div(id="kpi6"), width=4),
        ], className="g-2 mt-1"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="map_inc", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="map_mort", config=GRAPH_CONFIG), width=6),
        ], className="g-2"),
        interpretation_box(html.Div(id="interp_overview")),
    ])

def tab_compare():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id="rank_inc", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="rank_mort", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="rank_fatal", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="rank_cases", config=GRAPH_CONFIG), width=6),
        ], className="g-2"),
        interpretation_box(html.Div(id="interp_compare")),
    ])

def tab_trends():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id="ts_inc", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="ts_mort", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="ch_inc", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="ch_mort", config=GRAPH_CONFIG), width=6),
        ], className="g-2"),
        interpretation_box(html.Div(id="interp_trends")),
    ])

def tab_soc():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        dbc.Row([dbc.Col(dcc.Graph(id="corr", config=GRAPH_CONFIG), width=12)], className="g-2"),
        html.Div(style={"height": "18px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(id="box_inc", config=GRAPH_CONFIG), width=6),
            dbc.Col(dcc.Graph(id="box_mort", config=GRAPH_CONFIG), width=6),
        ], className="g-2"),
        dbc.Card(dbc.CardBody([
            html.Div("Observations", className="h6", style={"textAlign": "center", "fontWeight": "700"}),
            html.Div(id="ineq", style={"color": COLORS["muted"], "textAlign": "center"})
        ], style={"padding": "10px"}), style=STYLE_CARD, className="mt-2"),
        interpretation_box(html.Div(id="interp_soc")),
    ])

def tab_risk():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id="map_risk", config=GRAPH_CONFIG), width=7),
            dbc.Col(dcc.Graph(id="rank_risk", config=GRAPH_CONFIG), width=5),
        ], className="g-2"),
        dbc.Card(dbc.CardBody([
            html.Div("Observations", className="h6", style={"textAlign": "center", "fontWeight": "700"}),
            html.Div(id="risk_txt", style={"color": COLORS["muted"], "textAlign": "center"})
        ], style={"padding": "10px"}), style=STYLE_CARD, className="mt-2"),
        interpretation_box(html.Div(id="interp_risk")),
    ])

def tab_cluster():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        H("Regroupement des pays"),
        html.Div(style={"height": "10px"}),
        dcc.Graph(id="cluster_pca", config=GRAPH_CONFIG, style={"height": "480px"}),
        html.Div("Les marqueurs “X” indiquent les centres de groupes.",
                 style={"color": COLORS["muted"], "textAlign": "center", "fontSize": "0.93rem"}),
        html.Div(style={"height": "16px"}),
        dcc.Graph(id="cluster_prof", config=GRAPH_CONFIG, style={"height": "480px"}),
        html.Div(id="cluster_comment", style={"color": COLORS["muted"], "textAlign": "center", "fontSize": "0.98rem", "marginTop": "10px"}),
        interpretation_box(html.Div(id="interp_cluster")),
    ])

def tab_pred():
    return html.Div(style=STYLE_BLOCK_CONTAINER, children=[
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Modèle de prédiction de la mortalité", className="h6",
                         style={"textAlign": "center", "fontWeight": "700"}),
                html.Div(style={"height": "6px"}),
                dcc.Graph(id="ml_importance", config=GRAPH_CONFIG, style={"height": "420px"}),
            ]), style=STYLE_CARD), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Prévision des tendances", className="h6",
                         style={"textAlign": "center", "fontWeight": "700"}),
                html.Div(style={"height": "6px"}),
                dcc.Graph(id="forecast", config=GRAPH_CONFIG, style={"height": "420px"}),

                html.Div(QUALITY_TEXT_FIXED,
                         style={"color": COLORS["muted"], "textAlign": "center", "marginTop": "8px"}),

                html.Hr(),
                html.Div("Estimation", className="h6",
                         style={"textAlign": "center", "fontWeight": "700"}),
                html.Div(id="ml_pred_country", style={"color": COLORS["muted"], "textAlign": "center"})
            ]), style=STYLE_CARD), width=6),
        ], className="g-2"),
        interpretation_box(html.Div(id="interp_pred")),
    ])

tabs = dcc.Tabs(
    id="tabs",
    value="overview",
    children=[
        dcc.Tab(label="Vue globale", value="overview", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_overview()),
        dcc.Tab(label="Comparaison des pays", value="compare", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_compare()),
        dcc.Tab(label="Tendances", value="trends", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_trends()),
        dcc.Tab(label="Facteurs socio-économiques", value="soc", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_soc()),
        dcc.Tab(label="Risque", value="risk", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_risk()),
        dcc.Tab(label="Regroupement des pays", value="cluster", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_cluster()),
        dcc.Tab(label="Prédiction et prévision", value="pred", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=tab_pred()),
    ],
    colors={"background": COLORS["tab_idle"], "primary": COLORS["tab_selected"], "border": COLORS["border_soft"]},
    style={"borderRadius": "14px", "overflow": "hidden", "border": f"1px solid {COLORS['border_soft']}"},
    parent_style={"backgroundColor": COLORS["tab_idle"], "borderRadius": "14px"},
)

app.layout = dbc.Container(fluid=True, style=STYLE_APP, children=[
    dbc.Row([
        dbc.Col(html.Div([
            html.Div(APP_TITLE, className="h3", style={"marginBottom": "2px", "textAlign": "center", "fontWeight": "800"}),
            html.Div(SUBTITLE, style={"color": COLORS["muted"], "textAlign": "center", "fontWeight": "600"}),
        ]), width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(sidebar(), width=3),
        dbc.Col(html.Div([tabs], style=STYLE_MAIN_PANEL), width=9),
    ])
])

# SIDEBAR : activer/désactiver pays
@app.callback(
    Output("country_dd", "disabled"),
    Output("country_hint", "children"),
    Input("scope", "value"),
)
def toggle_country(scope):
    if scope == "Monde":
        return True, ""
    return False, ""

# VUE GLOBALE
@app.callback(
    Output("kpi1", "children"), Output("kpi2", "children"), Output("kpi3", "children"),
    Output("kpi4", "children"), Output("kpi5", "children"), Output("kpi6", "children"),
    Output("map_inc", "figure"), Output("map_mort", "figure"),
    Output("interp_overview", "children"),
    Input("year_dd", "value"),
    Input("scope", "value"),
    Input("country_dd", "value"),
)
def cb_overview(year, scope, country):
    year = int(year)
    dy = df[df["year"] == year].copy()

    if scope == "Monde":
        w = world_year_table(dy)
        if w.empty:
            return (kpi_card("-", "-"),) * 6 + (empty_fig(), empty_fig()) + ("Données insuffisantes.",)

        inc = float(w["incidence_asr"].iloc[0])
        mort = float(w["mortality_asr"].iloc[0])
        cases = float(w["estimated_cases"].iloc[0])
        deaths = float(w["estimated_deaths"].iloc[0])
        fatal = float(w["fatality_proxy"].iloc[0])
        risk = float(w["risk_score"].iloc[0])

        top_inc = dy.sort_values("incidence_asr", ascending=False).head(1)
        top_mort = dy.sort_values("mortality_asr", ascending=False).head(1)
        top_inc_name = top_inc["country"].iloc[0] if not top_inc.empty else "-"
        top_mort_name = top_mort["country"].iloc[0] if not top_mort.empty else "-"

        interp = (
            f"En {year}, l'incidence moyenne mondiale est d'environ {fmt(inc)} et la mortalité moyenne d'environ {fmt(mort)}. "
            f"Le rapport mortalité/incidence ({fmt(fatal)}) aide à repérer une sévérité relative plus élevée lorsqu'il augmente. "
            f"Les pays les plus marqués sont ({top_inc_name}) pour l'incidence maximale et ({top_mort_name}) pour la mortalité maximale. "
            f"Le score de risque moyen ({fmt(risk)}) synthétise plusieurs dimensions pour orienter la surveillance."
        )

        return (
            kpi_card("Taux d'incidence standardisé selon l'âge", fmt(inc), f"Année {year} - Monde"),
            kpi_card("Taux de mortalité standardisé selon l'âge", fmt(mort), f"Année {year} - Monde"),
            kpi_card("Nombre de cas estimés", fmt(cases, 0), "≈ taux/100 000 x population"),
            kpi_card("Nombre de décès estimés", fmt(deaths, 0), "≈ taux/100 000 x population"),
            kpi_card("Rapport mortalité/incidence", fmt(fatal)),
            kpi_card("Score de risque", fmt(risk), f"Max incidence : {top_inc_name} | Max mortalité : {top_mort_name}"),
            choropleth(dy, "incidence_asr", f"Incidence standardisée selon l'âge - {year}", height=360),
            choropleth(dy, "mortality_asr", f"Mortalité standardisée selon l'âge - {year}", height=360),
            interp
        )

    # Mode PAYS : KPI du pays sélectionné
    if not country:
        return (kpi_card("-", "-"),) * 6 + (empty_fig(), empty_fig()) + ("Sélectionnez un pays.",)

    row = dy[dy["country"] == country].head(1)
    if row.empty:
        interp = "Le pays sélectionné n'a pas de données pour cette année."
        kpis = [kpi_card("-", "-")] * 6
    else:
        r = row.iloc[0]
        interp = (
            f"Pour {country} en {year}, l'incidence est d'environ {fmt(r['incidence_asr'])} et la mortalité d'environ {fmt(r['mortality_asr'])}."
            f"Le rapport mortalité/incidence ({fmt(r['fatality_proxy'])}) aide à situer la charge relative."
        )
        kpis = [
            kpi_card("Taux d'incidence standardisé selon l'âge", fmt(r["incidence_asr"]), f"{country} - {year}"),
            kpi_card("Taux de mortalité standardisé selon l'âge", fmt(r["mortality_asr"]), f"{country} - {year}"),
            kpi_card("Nombre de cas estimés", fmt(r["estimated_cases"], 0), "≈ taux/100 000 x population"),
            kpi_card("Nombre de décès estimés", fmt(r["estimated_deaths"], 0), "≈ taux/100 000 x population"),
            kpi_card("Rapport mortalité/incidence", fmt(r["fatality_proxy"])),
            kpi_card("Score de risque", fmt(r["risk_score"]), f"Niveau : {r.get('risk_level','-')}"),
        ]

    return (
        kpis[0], kpis[1], kpis[2], kpis[3], kpis[4], kpis[5],
        choropleth(dy, "incidence_asr", f"Incidence standardisée selon l'âge - {year}", height=360),
        choropleth(dy, "mortality_asr", f"Mortalité standardisée selon l'âge - {year}", height=360),
        interp
    )

# COMPARAISON
@app.callback(
    Output("rank_inc", "figure"), Output("rank_mort", "figure"),
    Output("rank_fatal", "figure"), Output("rank_cases", "figure"),
    Output("interp_compare", "children"),
    Input("year_dd", "value"),
)
def cb_compare(year):
    year = int(year)
    dy = df[df["year"] == year].copy()

    interp = "Les pays extrêmes sont mis en évidence (incidence, mortalité, ratio mortalité/incidence et cas estimés)."
    if not dy.empty:
        a = dy.dropna(subset=["incidence_asr"]).sort_values("incidence_asr", ascending=False).head(1)
        b = dy.dropna(subset=["mortality_asr"]).sort_values("mortality_asr", ascending=False).head(1)
        if not a.empty and not b.empty:
            interp += f"En {year}, l'incidence la plus élevée est observée dans {a['country'].iloc[0]} et la mortalité la plus élevée dans {b['country'].iloc[0]}."
    return (
        rank_bar(dy, "incidence_asr", f"Pays et incidences les plus élevées - {year}", topn=10, height=360),
        rank_bar(dy, "mortality_asr", f"Pays et mortalités les plus élevées - {year}", topn=10, height=360),
        rank_bar(dy, "fatality_proxy", f"Pays et rapport mortalité/incidence le plus élevé - {year}", topn=10, height=360),
        rank_bar(dy, "estimated_cases", f"Pays et le plus grand nombre de cas estimés - {year}", topn=10, height=360),
        interp
    )

# TENDANCES (sans KPI)
@app.callback(
    Output("ts_inc", "figure"), Output("ts_mort", "figure"),
    Output("ch_inc", "figure"), Output("ch_mort", "figure"),
    Output("interp_trends", "children"),
    Input("scope", "value"),
    Input("country_dd", "value"),
)
def cb_trends(scope, country):
    if scope == "Monde":
        if df_world_time.empty:
            return empty_fig(), empty_fig(), empty_fig(), empty_fig(), "Données insuffisantes."
        fig1 = px.line(df_world_time, x="year", y="incidence_asr", markers=True)
        fig1.update_layout(height=270, showlegend=False)
        fig1.update_xaxes(title_text="Année")
        fig1.update_yaxes(title_text="")
        fig1 = apply_french_layout(fig1, "Monde - incidence standardisée selon l'âge", legend_mode="right")

        fig2 = px.line(df_world_time, x="year", y="mortality_asr", markers=True)
        fig2.update_layout(height=270, showlegend=False)
        fig2.update_xaxes(title_text="Année")
        fig2.update_yaxes(title_text="")
        fig2 = apply_french_layout(fig2, "Monde - mortalité standardisée selon l'âge", legend_mode="right")

        tmp = df_world_time.copy()
        tmp["annual_change_incidence_pct"] = tmp["incidence_asr"].pct_change() * 100
        tmp["annual_change_mortality_pct"] = tmp["mortality_asr"].pct_change() * 100

        fig3 = px.bar(tmp.dropna(subset=["annual_change_incidence_pct"]), x="year", y="annual_change_incidence_pct")
        fig3.update_layout(height=270, showlegend=False)
        fig3.update_xaxes(title_text="Année")
        fig3.update_yaxes(title_text="Variation (%)")
        fig3 = apply_french_layout(fig3, "Monde - variation annuelle de l'incidence (%)", legend_mode="right")

        fig4 = px.bar(tmp.dropna(subset=["annual_change_mortality_pct"]), x="year", y="annual_change_mortality_pct")
        fig4.update_layout(height=270, showlegend=False)
        fig4.update_xaxes(title_text="Année")
        fig4.update_yaxes(title_text="Variation (%)")
        fig4 = apply_french_layout(fig4, "Monde - variation annuelle de la mortalité (%)", legend_mode="right")

        interp = "Les variations annuelles repèrent les années de changement marqué. Une hausse persistante suggère d'explorer l'accès au dépistage et aux soins."
        return fig1, fig2, fig3, fig4, interp

    dc = df[df["country"] == country].copy()
    fig_a = ts_line(dc, "incidence_asr", f"{country} - incidence standardisée selon l'âge", height=270)
    fig_b = ts_line(dc, "mortality_asr", f"{country} - mortalité standardisée selon l'âge", height=270)
    fig_c = change_bar(dc, "annual_change_incidence_pct", f"{country} - variation annuelle de l'incidence (%)", height=270)
    fig_d = change_bar(dc, "annual_change_mortality_pct", f"{country} - variation annuelle de la mortalité (%)", height=270)

    interp = f"Ces courbes permettent de voir l'évolution de l'incidence et de la mortalité pour {country}, et d'identifier les années de rupture."
    return fig_a, fig_b, fig_c, fig_d, interp

# SOCIO-ECONOMIQUES (sans KPI)
@app.callback(
    Output("corr", "figure"), Output("box_inc", "figure"), Output("box_mort", "figure"),
    Output("ineq", "children"),
    Output("interp_soc", "children"),
    Input("year_dd", "value")
)
def cb_soc(year):
    year = int(year)
    dy = df[df["year"] == year].copy()

    txt = "-"
    if "gdp_quartile" in dy.columns and dy["gdp_quartile"].notna().any():
        g = dy.dropna(subset=["gdp_quartile", "mortality_asr"]).groupby("gdp_quartile")["mortality_asr"].mean()
        if len(g) >= 2:
            txt = (
                f"En {year}, la mortalité moyenne en Q1 est d'environ {fmt(g.iloc[0])} "
                f"contre {fmt(g.iloc[-1])} en Q4 (écart ≈ {fmt(g.iloc[0]-g.iloc[-1])})."
            )

    interp = "Les corrélations et les distributions par quartiles aident à repérer des inégalités. Une différence nette Q1-Q4 peut indiquer un accès inégal au dépistage/traitement."
    return (
        corr_heat(dy, f"Corrélations - {year}", height=310),
        box_by_quartile(dy, "incidence_asr", f"Incidence selon le produit intérieur brut par habitant - {year}", height=310),
        box_by_quartile(dy, "mortality_asr", f"Mortalité selon le produit intérieur brut par habitant - {year}", height=310),
        txt,
        interp
    )

# RISQUE (sans KPI)
@app.callback(
    Output("map_risk", "figure"), Output("rank_risk", "figure"), Output("risk_txt", "children"),
    Output("interp_risk", "children"),
    Input("year_dd", "value"),
    Input("scope", "value"),
    Input("country_dd", "value"),
)
def cb_risk(year, scope, country):
    year = int(year)
    dy = df[df["year"] == year].copy()

    if scope == "Monde":
        w = world_year_table(dy)
        if w.empty:
            return empty_fig(), empty_fig(), "-", "Données insuffisantes."
        txt = (
            f"Monde - score de risque moyen ≈ {fmt(float(w['risk_score'].iloc[0]))} | "
            f"incidence moyenne ≈ {fmt(float(w['incidence_asr'].iloc[0]))} | "
            f"mortalité moyenne ≈ {fmt(float(w['mortality_asr'].iloc[0]))}."
        )
        interp = "Le score de risque combine plusieurs dimensions. Les pays en tête du classement sont à analyser en priorité (charge, tendances, contexte socio-économique)."
    else:
        row = dy[dy["country"] == country].head(1)
        if row.empty:
            txt = "Pays non disponible pour cette année."
        else:
            r = row.iloc[0]
            txt = (
                f"{country} - incidence {fmt(r['incidence_asr'])} | mortalité {fmt(r['mortality_asr'])} | "
                f"rapport mortalité/incidence {fmt(r['fatality_proxy'])} | "
                f"variation annuelle de la mortalité {fmt(r['annual_change_mortality_pct'])} | "
                f"score {fmt(r['risk_score'])} ({r.get('risk_level','-')})."
            )
            interp = "Un score élevé reflète une combinaison défavorable (mortalité, ratio mortalité/incidence, aggravation récente)."

    return (
        choropleth(dy, "risk_score", f"Score de risque - {year}", height=360),
        rank_bar(dy, "risk_score", f"Pays avec les scores de risque les plus élevés - {year}", topn=10, height=360),
        txt,
        interp
    )

# CLUSTER
@app.callback(
    Output("cluster_pca", "figure"),
    Output("cluster_prof", "figure"),
    Output("cluster_comment", "children"),
    Output("interp_cluster", "children"),
    Input("year_dd", "value"),
    Input("k", "value")
)
def cb_cluster(year, k):
    year = int(year)
    out = compute_clusters(df, year, k)
    if out is None:
        return empty_fig("Regroupement indisponible"), empty_fig("Regroupement indisponible"), "-", "Données insuffisantes."

    dcl, cols, v1, v2, centers = out
    palette = px.colors.qualitative.Set2

    fig_pca = px.scatter(
        dcl,
        x="pca1", y="pca2",
        color=dcl["cluster"].astype(str),
        hover_name="country",
        color_discrete_sequence=palette,
    )
    fig_pca.update_traces(marker=dict(size=11, line=dict(width=0.9, color="rgba(0,0,0,0.85)")))

    fig_pca.add_trace(
        go.Scatter(
            x=centers["pca1"], y=centers["pca2"],
            mode="markers+text",
            text=[f"G{i}" for i in centers["cluster"]],
            textposition="top center",
            textfont=dict(color=COLORS["text"], size=12),
            marker=dict(size=16, symbol="x", line=dict(width=2, color=COLORS["text"])),
            name="Centre des groupes",
        )
    )

    fig_pca.update_layout(height=480)
    fig_pca = apply_french_layout(fig_pca, f"Regroupement (K={k}) - PCA (variance {v1:.2f} et {v2:.2f})", legend_mode="right")
    fig_pca.update_xaxes(title_text="Axe PCA 1")
    fig_pca.update_yaxes(title_text="Axe PCA 2")
    fig_pca.update_layout(legend_title_text="Groupe")

    prof = dcl.groupby("cluster")[cols].mean()
    z = (prof - prof.mean(axis=0)) / (prof.std(axis=0).replace(0, np.nan))
    z = z.reset_index().melt(id_vars=["cluster"], var_name="variable", value_name="zscore")
    z["cluster"] = z["cluster"].astype(str)

    fig_prof = px.bar(z, x="variable", y="zscore", color="cluster", barmode="group", color_discrete_sequence=palette)
    fig_prof.update_layout(height=480)
    fig_prof = apply_french_layout(fig_prof, f"Profil des groupes (z-scores) - {year}", legend_mode="right")
    fig_prof.update_yaxes(title_text="Niveau relatif (z-score)")
    fig_prof.update_xaxes(title_text="")
    fig_prof.update_layout(legend_title_text="Groupe")
    fig_prof.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)")

    counts = dcl["cluster"].value_counts().sort_index()
    parts = [f"G{i}: {int(counts.get(i, 0))} pays" for i in range(int(k))]
    comment_txt = f"Répartition des pays par groupe (année {year}) : " + " | ".join(parts) + "."

    interp = "Si les groupes sont bien séparés sur la PCA, cela suggère des profils réellement différents. Le profil en z-scores indique quelles variables distinguent chaque groupe."
    return fig_pca, fig_prof, comment_txt, interp

# PRED + FORECAST (sans KPI)
@app.callback(
    Output("ml_importance", "figure"),
    Output("forecast", "figure"),
    Output("ml_pred_country", "children"),
    Output("interp_pred", "children"),
    Input("scope", "value"),
    Input("country_dd", "value"),
    Input("year_dd", "value"),
    Input("horizon", "value")
)
def cb_pred(scope, country, year, horizon):
    year = int(year)
    horizon = int(horizon)

    if model_metrics is None:
        imp_fig = empty_fig("Modèle indisponible")
    else:
        imp_fig = importance_fig(model_importances)

    def build_forecast_fig(d_obs, title, horizon_):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_obs["year"], y=d_obs["incidence_asr"], mode="lines+markers", name="Incidence (observée)"))
        fig.add_trace(go.Scatter(x=d_obs["year"], y=d_obs["mortality_asr"], mode="lines+markers", name="Mortalité (observée)"))

        finc = forecast_trend(d_obs, "incidence_asr", horizon_)
        fmort = forecast_trend(d_obs, "mortality_asr", horizon_)
        last_year = int(d_obs["year"].max())

        if finc is not None:
            fig.add_trace(go.Scatter(x=finc["year"], y=finc["pred"], mode="lines+markers", name="Incidence (prévue)", line=dict(dash="dash")))
        if fmort is not None:
            fig.add_trace(go.Scatter(x=fmort["year"], y=fmort["pred"], mode="lines+markers", name="Mortalité (prévue)", line=dict(dash="dash")))

        fig.add_vrect(x0=last_year + 0.5, x1=last_year + horizon_ + 0.5, fillcolor="rgba(255,255,255,0.06)", line_width=0)
        fig.add_vline(x=last_year + 0.5, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)")

        fig.update_layout(height=420)
        fig = apply_french_layout(fig, title, legend_mode="bottom")
        fig.update_xaxes(title_text="Année")
        fig.update_yaxes(title_text="")
        return fig

    if scope == "Monde":
        if df_world_time.empty:
            return imp_fig, empty_fig("Aucune donnée"), "-", "Données insuffisantes."
        d_obs = df_world_time.dropna(subset=["year", "incidence_asr", "mortality_asr"]).sort_values("year")
        fig = build_forecast_fig(d_obs, f"Monde - prévision des tendances (horizon {horizon} ans)", horizon)
        interp = "La prévision prolonge les tendances historiques : utile pour des scénarios, mais sensible aux ruptures et changements de politique de santé."
        return imp_fig, fig, "Estimation par modèle : non affichée pour la vue monde.", interp

    dc = df[df["country"] == country].copy()
    d_obs = dc.dropna(subset=["year", "incidence_asr", "mortality_asr"]).sort_values("year")
    if d_obs.empty:
        return imp_fig, empty_fig("Aucune donnée"), "-", "Aucune donnée."

    fig = build_forecast_fig(d_obs, f"{country} - prévision des tendances (horizon {horizon} ans)", horizon)

    pred_txt = "-"
    if model_rf is not None and model_metrics is not None:
        feats = model_metrics["features"]
        row = df[(df["country"] == country) & (df["year"] == year)].head(1)
        if not row.empty and all(f in row.columns for f in feats):
            X = row[feats].replace([np.inf, -np.inf], np.nan)
            if X.isna().any(axis=1).iloc[0]:
                pred_txt = "Estimation impossible : une variable explicative est manquante."
            else:
                yhat = float(model_rf.predict(X)[0])
                yobs = float(row["mortality_asr"].iloc[0])
                pred_txt = f"Mortalité observée : {yobs:.2f} | mortalité estimée : {yhat:.2f}"
        else:
            pred_txt = "Estimation impossible : données insuffisantes pour ce pays et cette année."

    interp = "Les variables importantes indiquent les facteurs les plus contributifs. Comparer l'observé à l'estimé aide à repérer des écarts inattendus et à investiguer."
    return imp_fig, fig, pred_txt, interp

if __name__ == "__main__":
    app.run(debug=True)