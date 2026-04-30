# Trading_Architecture.py
# Batch multi-actions : régression 10 ans + RSI + recherche par nom (EODHD)
# ---------------------------------------------------------------------------
# Prérequis :
#   pip install streamlit pandas numpy requests matplotlib
#   Exporter la clé API :
#       Linux/Mac :  export EODHD_API_KEY="TA_CLE"
#       Windows   :  set EODHD_API_KEY=TA_CLE
#   Lancer :
#       streamlit run Trading_Architecture.py
# ---------------------------------------------------------------------------

import os
import time
import math
import datetime as dt
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------
EODHD_API = "https://eodhd.com/api"
API_KEY = os.getenv("EODHD_API_KEY", "")
REQUEST_SLEEP = 0.25
TIMEOUT = 25


# ---------------------------------------------------------------------------
# Helpers API
# ---------------------------------------------------------------------------
def _to_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def _req(url: str, params: dict, sleep: float = REQUEST_SLEEP):
    p = {**params, "api_token": API_KEY, "fmt": "json"}
    r = requests.get(url, params=p, timeout=TIMEOUT)
    if sleep:
        time.sleep(sleep)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_eod_daily(symbol: str, exchange: str, years: int = 10) -> pd.DataFrame:
    """
    Récupère l'historique quotidien sur ~years+1 ans et construit une colonne 'price'
    en privilégiant 'adjusted_close' (corrigé des splits), puis 'close' si besoin.
    """
    since = _to_date(dt.date.today() - dt.timedelta(days=365 * (years + 1)))
    url = f"{EODHD_API}/eod/{symbol}.{exchange}"

    try:
        data = _req(url, {"from": since, "to": _to_date(dt.date.today())})
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Choix de la colonne de prix la plus fiable
    price_col = None
    if "adjusted_close" in df.columns:
        price_col = "adjusted_close"
    elif "close" in df.columns:
        price_col = "close"
    else:
        return pd.DataFrame()

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price"])

    return df


@st.cache_data(show_spinner=False, ttl=1800)
def search_instruments(query: str, limit: int = 50) -> pd.DataFrame:
    """
    Recherche par nom/ISIN/symbole via l'API de recherche EODHD.
    On essaie ensuite de mettre en avant la "cotation principale" :
    - priorité aux exchanges classiques (US, PA, DE, LSE, MI…)
    - boost si le nom ou le ticker matche bien la requête
    - la meilleure ligne reçoit un ⭐ dans la liste
    """
    q = query.strip()
    if not q:
        return pd.DataFrame()

    url = f"{EODHD_API}/search/{q}"
    try:
        data = _req(url, {"limit": limit}, sleep=0.0)
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Harmonisation colonnes
    cols_map = {
        "Name": "Name", "name": "Name",
        "Code": "Code", "code": "Code",
        "Exchange": "Exchange", "exchange": "Exchange",
        "Country": "Country", "country": "Country",
        "Type": "Type", "type": "Type",
    }
    for k, v in list(cols_map.items()):
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    keep = [c for c in ["Name", "Code", "Exchange", "Country", "Type"] if c in df.columns]
    df = df[keep].dropna(subset=["Code", "Exchange"])

    # Scoring pour repérer la cotation principale
    q_low = q.lower()

    exchange_priority = [
        "US", "NASDAQ", "NYSE", "NMS", "NYS",
        "PA", "XPAR",
        "DE", "XETRA", "F",
        "LSE", "L",
        "MI", "BIT",
        "AS", "AMS",
        "BRU", "BR",
        "ES", "BM",
    ]

    def score_row(r):
        name = str(r.get("Name", "")).lower()
        code = str(r.get("Code", "")).lower()
        exch = str(r.get("Exchange", "")).upper()
        s = 0

        # Match du nom
        if q_low == name:
            s += 5
        elif q_low in name:
            s += 3
        elif name.startswith(q_low):
            s += 2

        # Match du ticker
        if q_low == code:
            s += 4
        elif q_low in code:
            s += 2

        # Priorité de la place de cotation
        if exch in exchange_priority:
            s += 5 + (len(exchange_priority) - exchange_priority.index(exch)) * 0.1

        # Bonus si type "Common Stock"
        t = str(r.get("Type", "")).lower()
        if "common" in t or "stock" in t:
            s += 1

        return s

    df["score"] = df.apply(score_row, axis=1)

    # Déduplication & tri
    df["key"] = df["Code"].astype(str) + "." + df["Exchange"].astype(str)
    df = df.sort_values("score", ascending=False).drop_duplicates("key")

    # Label avec ⭐ pour la meilleure ligne
    max_score = df["score"].max() if not df["score"].empty else 0
    labels = []
    for _, r in df.iterrows():
        prefix = "⭐ " if r["score"] == max_score and max_score > 0 else ""
        labels.append(f"{prefix}{r['Name']} — {r['Code']}.{r['Exchange']}")

    df["Label"] = labels

    return df


# ---------------------------------------------------------------------------
# Calculs techniques : resampling, RSI, régression
# ---------------------------------------------------------------------------
def resample_weekly_monthly(price: pd.Series):
    price_w = price.resample("W-FRI").last().dropna()
    price_m = price.resample("ME").last().dropna()
    return price_w, price_m


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def linear_regression_sigma(price_w: pd.Series):
    x = np.arange(len(price_w), dtype=float)
    y = price_w.values.astype(float)
    if len(y) < 3:
        idx = price_w.index
        return pd.Series(index=idx, dtype=float), np.nan, pd.Series(index=idx, dtype=float)

    a, b = np.polyfit(x, y, 1)  # y = a*x + b
    y_hat = a * x + b
    resid = y - y_hat
    sigma = np.std(resid, ddof=1) if len(resid) > 2 else np.nan
    z = np.where(sigma and not math.isnan(sigma) and sigma > 0, resid / sigma, np.nan)
    return pd.Series(y_hat, index=price_w.index), float(sigma), pd.Series(z, index=price_w.index)


def analyze_one(
    symbol: str,
    exchange: str,
    years: int,
    show_rsi: bool,
    rsi_period: int,
    display_name: Optional[str] = None,
) -> Optional[Dict]:
    df = fetch_eod_daily(symbol, exchange, years=years)
    if df.empty or "price" not in df.columns:
        st.warning(f"❌ {symbol}.{exchange} : données introuvables ou inexploitables.")
        return None

    price = df["price"].astype(float)
    price_w, price_m = resample_weekly_monthly(price)

    # Fenêtre de régression sur les X dernières années
    cutoff = price_w.index[-1] - pd.DateOffset(years=years)
    pw = price_w[price_w.index >= cutoff]
    if len(pw) < 150:
        st.warning(f"⚠️ {symbol}.{exchange} : historique hebdo insuffisant pour {years} ans.")
        return None

    reg, sigma, z = linear_regression_sigma(pw)
    last_price = pw.iloc[-1]
    last_reg = reg.iloc[-1] if len(reg) else np.nan
    last_z = z.iloc[-1] if len(z) else np.nan
    reg_p25 = last_reg + 2.5 * sigma if (sigma and not math.isnan(sigma)) else np.nan
    reg_m25 = last_reg - 2.5 * sigma if (sigma and not math.isnan(sigma)) else np.nan

    rsi_w_val = rsi_m_val = None
    if show_rsi:
        try:
            rsi_w = compute_rsi(price_w, rsi_period)
            rsi_m = compute_rsi(price_m, rsi_period)
            rsi_w_val = float(rsi_w.iloc[-1]) if len(rsi_w) else None
            rsi_m_val = float(rsi_m.iloc[-1]) if len(rsi_m) else None
        except Exception:
            pass

    title = f"{symbol}.{exchange}" if not display_name else f"{display_name} — {symbol}.{exchange}"

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(pw.index, pw.values, label="Prix (hebdo)")
    ax.plot(reg.index, reg.values, label="Régression")
    if sigma and not math.isnan(sigma):
        ax.plot(reg.index, reg.values + 2.5 * sigma, label="+2.5σ", linestyle="--")
        ax.plot(reg.index, reg.values - 2.5 * sigma, label="-2.5σ", linestyle="--")
    ax.set_title(f"{title} — Régression {years} ans (hebdo)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.legend(loc="best")

    metrics = {
        "Cours": f"{last_price:,.2f}",
        "Régression": f"{last_reg:,.2f}",
        "Reg +2.5σ": (f"{reg_p25:,.2f}" if not math.isnan(reg_p25) else "n/a"),
        "Reg -2.5σ": (f"{reg_m25:,.2f}" if not math.isnan(reg_m25) else "n/a"),
        "z-score (σ)": (f"{last_z:,.2f}" if not math.isnan(last_z) else "n/a"),
    }

    row = {
        "name": display_name if display_name else "",
        "symbol": symbol,
        "exchange": exchange,
        "price": round(float(last_price), 4),
        "reg": round(float(last_reg), 4) if not math.isnan(last_reg) else np.nan,
        "reg+2.5σ": round(float(reg_p25), 4) if not math.isnan(reg_p25) else np.nan,
        "reg-2.5σ": round(float(reg_m25), 4) if not math.isnan(reg_m25) else np.nan,
        "z_score_σ": round(float(last_z), 2) if not math.isnan(last_z) else np.nan,
        "rsi_weekly": round(float(rsi_w_val), 2) if rsi_w_val is not None else np.nan,
        "rsi_monthly": round(float(rsi_m_val), 2) if rsi_m_val is not None else np.nan,
        "asof": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    return {"fig": fig, "metrics": metrics, "title": title, "row": row}


# ---------------------------------------------------------------------------
# UI Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Régression 10 ans — Batch multi-actions", layout="wide")
st.title("📉 Régression linéaire 10 ans — Batch multi-actions (EODHD)")

with st.expander("ℹ️ Mode d’emploi", expanded=False):
    st.markdown(
        """
- Ajoute des actions de deux manières (tu peux combiner) :
  1. Zone **`TICKER,EXCHANGE`** (multi-lignes) — ex. `AAPL,US` • `MSFT,US` • `DSY,PA`
  2. **Recherche par nom** → multisélection → elles seront ajoutées à la liste à analyser.
- Pour chaque action, l’app calcule :
  **prix hebdo ~10 ans**, **régression**, **σ**, **z-score**, **bandes ±2,5σ**, **RSI Weekly/Monthly (option)**.
        """
    )

if not API_KEY:
    st.error("⚠️ Renseigne la variable d’environnement **EODHD_API_KEY** avant de lancer l’app.")
    st.stop()

st.sidebar.header("Paramètres")
years = st.sidebar.number_input("Fenêtre historique (années)", value=10, min_value=5, max_value=20, step=1)
show_rsi = st.sidebar.checkbox("Afficher RSI Weekly / Monthly", value=True)
rsi_period = st.sidebar.number_input("Période RSI", value=14, step=1)
display_mode = st.sidebar.selectbox("Affichage des graphiques", ["Pile", "Onglets", "Grille 2 colonnes"])

# État pour l'entrée manuelle & ajouts via recherche
if "manual_lines" not in st.session_state:
    st.session_state["manual_lines"] = "AAPL,US"

if "search_added_lines" not in st.session_state:
    st.session_state["search_added_lines"] = []

# ---- Entrée 1 : TICKER,EXCHANGE
st.subheader("➊ Entrée manuelle : `TICKER,EXCHANGE`")
st.caption("Exemples : `AAPL,US` • `MSFT,US` • `DSY,PA` • `SAP,DE`")

tickers_text = st.text_area(
    "Colle ou remplis tes lignes ici :",
    value=st.session_state["manual_lines"],
    height=120,
)
st.session_state["manual_lines"] = tickers_text

# ---- Entrée 2 : Recherche par nom
st.subheader("➋ Recherche par nom")
name_query = st.text_input("Nom d’entreprise / ISIN / symbole", value="")
search_clicked = st.button("🔎 Rechercher")

if "search_df" not in st.session_state:
    st.session_state["search_df"] = pd.DataFrame()

if search_clicked and name_query.strip():
    with st.spinner("Recherche en cours…"):
        search_df = search_instruments(name_query.strip(), limit=50)
    st.session_state["search_df"] = search_df
    if search_df.empty:
        st.warning("Aucun résultat.")

search_df = st.session_state["search_df"]

if not search_df.empty:
    st.write("Résultats de recherche :")
    labels = search_df["Label"].tolist()
    selected_labels = st.multiselect(
        "Sélectionne une ou plusieurs sociétés :",
        labels,
        key="search_multiselect",
    )

    add_clicked = st.button("➕ Ajouter ces actions à la sélection")
    if add_clicked and selected_labels:
        new_lines = []
        for lab in selected_labels:
            row = search_df[search_df["Label"] == lab].iloc[0]
            code = row["Code"]
            exch = row["Exchange"]
            line = f"{code},{exch}"
            if line not in st.session_state["search_added_lines"]:
                st.session_state["search_added_lines"].append(line)
                new_lines.append(line)

        st.success(
            f"{len(new_lines)} action(s) ajoutée(s) à la liste à analyser."
        )

if st.session_state["search_added_lines"]:
    st.markdown("**Actions ajoutées via la recherche (en plus de l’entrée manuelle) :**")
    st.code("\n".join(st.session_state["search_added_lines"]))

run = st.button("🚀 Lancer l’analyse")

# ---------------------------------------------------------------------------
# Lancement de l'analyse
# ---------------------------------------------------------------------------
if run:
    # 1) Lignes manuelles
    lines = [l.strip() for l in st.session_state["manual_lines"].splitlines() if l.strip()]

    # 2) Ajouts via recherche
    lines.extend(st.session_state["search_added_lines"])

    # Déduplication
    lines = list(dict.fromkeys(lines))

    pairs: List[Tuple[str, str, str]] = []  # (Name(optional), Symbol, Exchange)

    for l in lines:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            pairs.append(("", parts[0], parts[1]))
        else:
            st.warning(f"Ligne ignorée (format attendu TICKER,EXCHANGE) : {l}")

    if not pairs:
        st.error("Aucune action valide à analyser.")
    else:
        results_rows = []
        cards: List[Dict] = []

        progress = st.progress(0)
        done = 0
        total = len(pairs)

        for display_name, sym, exch in pairs:
            res = analyze_one(
                sym,
                exch,
                years=years,
                show_rsi=show_rsi,
                rsi_period=rsi_period,
                display_name=display_name or None,
            )
            if res:
                cards.append(res)
                results_rows.append(res["row"])
            done += 1
            progress.progress(min(1.0, done / total))

        progress.empty()

        if not cards:
            st.warning("Aucun résultat exploitable.")
        else:
            # Affichage des graphiques
            if display_mode == "Onglets":
                tabs = st.tabs([c["title"] for c in cards])
                for tab, c in zip(tabs, cards):
                    with tab:
                        st.pyplot(c["fig"], clear_figure=True)
                        m = c["metrics"]
                        colz = st.columns(5)
                        colz[0].metric("Cours", m["Cours"])
                        colz[1].metric("Régression", m["Régression"])
                        colz[2].metric("Reg +2.5σ", m["Reg +2.5σ"])
                        colz[3].metric("Reg -2.5σ", m["Reg -2.5σ"])
                        colz[4].metric("z-score (σ)", m["z-score (σ)"])
                        st.markdown("---")
            elif display_mode == "Grille 2 colonnes":
                for i in range(0, len(cards), 2):
                    c1, c2 = st.columns(2)
                    for col, c in zip([c1, c2], cards[i : i + 2]):
                        with col:
                            st.pyplot(c["fig"], clear_figure=True)
                            m = c["metrics"]
                            colz = st.columns(5)
                            colz[0].metric("Cours", m["Cours"])
                            colz[1].metric("Régression", m["Régression"])
                            colz[2].metric("Reg +2.5σ", m["Reg +2.5σ"])
                            colz[3].metric("Reg -2.5σ", m["Reg -2.5σ"])
                            colz[4].metric("z-score (σ)", m["z-score (σ)"])
                    st.markdown("---")
            else:  # Pile
                for c in cards:
                    st.pyplot(c["fig"], clear_figure=True)
                    m = c["metrics"]
                    colz = st.columns(5)
                    colz[0].metric("Cours", m["Cours"])
                    colz[1].metric("Régression", m["Régression"])
                    colz[2].metric("Reg +2.5σ", m["Reg +2.5σ"])
                    colz[3].metric("Reg -2.5σ", m["Reg -2.5σ"])
                    colz[4].metric("z-score (σ)", m["z-score (σ)"])
                    st.markdown("---")

            # Export CSV
            if results_rows:
                df_out = pd.DataFrame(results_rows)
                st.markdown("### 📄 Export des niveaux de régression")
                st.dataframe(df_out, use_container_width=True, height=350)
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Télécharger CSV",
                    csv,
                    file_name="regression_levels_batch.csv",
                    mime="text/csv",
                )
