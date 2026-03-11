"""
Screener de Activos de Calidad + Asistente IA
==============================================
Filtra empresas de calidad con la base de Morningstar
y permite hacer preguntas sobre los resultados usando
Claude (Anthropic) o GPT-4o (OpenAI).

Instalación:
    pip install streamlit anthropic openai

Ejecución:
    streamlit run screener_chat_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Screener IA – Morningstar",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────
# CARGA Y PROCESAMIENTO DE DATOS (cached)
# ──────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    # ── Morningstar: calidad fundamental ──
    df = pd.read_csv(BASE_DIR / "bbdd_morningstar.csv")

    # Puntaje de calidad (0-100)
    df["pts_rating"]     = (df["QuantitativeStarRating"] >= 4).astype(int) * 20
    df["pts_roe"]        = (df["ROEYear1"]       > 0.15).astype(int) * 15
    df["pts_roic"]       = (df["ROICYear1"]      > 0.10).astype(int) * 15
    df["pts_margen"]     = (df["NetMargin"]       > 0.10).astype(int) * 15
    df["pts_deuda"]      = (df["DebtEquityRatio"] < 1.50).astype(int) * 15
    df["pts_eps"]        = (df["EPSGrowth3YYear1"] > 0).astype(int) * 10
    df["pts_revenue"]    = (df["RevenueGrowth3Y"]  > 5).astype(int) * 10

    cols_pts = ["pts_rating","pts_roe","pts_roic","pts_margen",
                "pts_deuda","pts_eps","pts_revenue"]

    metricas = ["QuantitativeStarRating","ROEYear1","ROICYear1",
                "NetMargin","EBTMarginYear1","DebtEquityRatio",
                "EPSGrowth3YYear1","RevenueGrowth3Y"]

    df["datos_ok"] = df[metricas].notna().sum(axis=1)
    df["puntaje"]  = df[cols_pts].sum(axis=1)
    df.loc[df["datos_ok"] < 4, "puntaje"] = np.nan

    # ── Yahoo Finance: señales de corto plazo ──
    YAHOO_PATH = BASE_DIR / "bbdd_yahoo.csv"
    try:
        _want = {"ticker","currentPrice","fiftyDayAverage","twoHundredDayAverage",
                 "volume","averageDailyVolume10Day",
                 "fiftyTwoWeekLowChangePercent","fiftyTwoWeekHighChangePercent","beta"}
        yf = pd.read_csv(YAHOO_PATH, usecols=lambda c: c in _want)
        yf["ticker"] = yf["ticker"].str.upper()
        yf = yf.rename(columns={"ticker": "Ticker"})

        # Calcular señales derivadas (todas en %)
        yf["ma50_pct"]   = ((yf["currentPrice"] / yf["fiftyDayAverage"])   - 1) * 100
        yf["ma200_pct"]  = ((yf["currentPrice"] / yf["twoHundredDayAverage"]) - 1) * 100
        yf["vol_ratio"]  = yf["volume"] / yf["averageDailyVolume10Day"]
        # yfinance devuelve fiftyTwoWeekLowChangePercent en decimal → convertir a %
        yf["dist_52w_low"]  = yf["fiftyTwoWeekLowChangePercent"]  * 100
        yf["dist_52w_high"] = yf["fiftyTwoWeekHighChangePercent"] * 100

        _yf_cols = ["Ticker","ma50_pct","ma200_pct","vol_ratio",
                    "dist_52w_low","dist_52w_high","beta"]
        df = df.merge(yf[_yf_cols], on="Ticker", how="left")
    except Exception:
        pass  # Si bbdd_yahoo.csv no existe, continúa sin datos de corto plazo

    return df

# ──────────────────────────────────────────────
# SCRAPING DE MORNINGSTAR (igual que clase_07)
# ──────────────────────────────────────────────
CSV_PATH = BASE_DIR / "bbdd_morningstar.csv"

FEATURES = [
    'ClosePrice','DebtEquityRatio','DividendYield','EBTMarginYear1',
    'EPSGrowth3YYear1','EquityStyleBox','IndustryName','MarketCap',
    'MarketCountryName','Name','NetMargin','PEGRatio','PERatio',
    'QuantitativeStarRating','ROATTM','ROETTM','ROEYear1','ROICYear1',
    'ReturnD1','ReturnM0','ReturnM1','ReturnM12','ReturnM120','ReturnM3',
    'ReturnM36','ReturnM6','ReturnM60','ReturnW1','RevenueGrowth3Y',
    'SectorName','Ticker','Universe',
]

EXCHANGES = {
    "E0EXG$XNYS": "NYSE",
    "E0EXG$XNAS": "Nasdaq",
    "E0EXG$XLON": "Londres",
    "E0EXG$XSHG": "Shangai",
    "E0EXG$XHKG": "Hong Kong",
    "E0EXG$XPAR": "Paris",
    "E0EXG$XAMS": "Amsterdam",
    "E0EXG$XTKS": "Tokyo",
    "E0EXG$XASX": "Australian Securities Exchange",
    "E0EXG$XSWX": "Swiss Exchange",
    "E0EXG$XTSE": "Toronto",
    "E0EXG$XBUE": "Argentina",
}

def actualizar_datos_morningstar(exchanges_sel: list, progress_bar, status_text) -> tuple[bool, str]:
    """
    Descarga datos frescos de Morningstar para los exchanges elegidos
    y sobreescribe el CSV local. Devuelve (éxito, mensaje).
    """
    datos = []
    total = len(exchanges_sel)

    for i, code in enumerate(exchanges_sel):
        nombre = EXCHANGES[code]
        status_text.text(f"⬇️ Descargando {nombre}… ({i+1}/{total})")
        try:
            params = {
                'page': '1',
                'pageSize': '50000',
                'sortOrder': 'Name asc',
                'outputType': 'json',
                'version': '1',
                'languageId': 'en-GB',
                'currencyId': 'USD',
                'universeIds': code,
                'securityDataPoints': '|'.join(FEATURES),
                'filters': '',
                'term': '',
                'subUniverseId': '',
            }
            resp = requests.get(
                'https://tools.morningstar.co.uk/api/rest.svc/klr5zyak8x/security/screener',
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            filas = resp.json().get('rows', [])
            df_exchange = pd.DataFrame(filas)
            df_exchange['ExchangeName'] = nombre
            datos.append(df_exchange)
        except Exception as e:
            status_text.text(f"⚠️ Error en {nombre}: {e}")

        progress_bar.progress((i + 1) / total)

    if not datos:
        return False, "No se pudieron descargar datos de ningún exchange."

    df_nuevo = pd.concat(datos, ignore_index=True)
    df_nuevo.to_csv(CSV_PATH, index=False)
    return True, f"✅ {len(df_nuevo):,} empresas descargadas de {len(datos)} exchanges."


def fecha_actualizacion() -> str:
    """Devuelve la fecha de última modificación del CSV."""
    if os.path.exists(CSV_PATH):
        ts = os.path.getmtime(CSV_PATH)
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
    return "Desconocida"


# ──────────────────────────────────────────────
# PERSISTENCIA DE CONFIGURACIÓN
# ──────────────────────────────────────────────
SETTINGS_PATH = BASE_DIR / "screener_settings.json"

_DEFAULTS_CFG = {
    "cfg_proveedor":         "Claude (Anthropic)",
    "cfg_puntaje_min":       60,
    "cfg_sector_sel":        [],
    "cfg_mcap_min":          1.0,
    "cfg_pe_max":            40,
    "cfg_div_min":           0.0,
    "cfg_ma50_on":           False,
    "cfg_ma200_on":          False,
    "cfg_vol_on":            False,
    "cfg_ret_w1_min":        0.0,
    "cfg_ret_m1_min":        0.0,
    "cfg_exchanges_nombres": ["NYSE", "Nasdaq"],
    # Búsqueda semántica
    "sem_tickers":           None,
    "sem_query_prev":        "",
}

def cargar_config() -> dict:
    """Carga configuración guardada. Devuelve defaults si no existe o hay error."""
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            merged = {**_DEFAULTS_CFG, **saved}
            # Asegurar tipos correctos para evitar errores en widgets
            merged["cfg_puntaje_min"]  = int(merged["cfg_puntaje_min"])
            merged["cfg_mcap_min"]     = float(merged["cfg_mcap_min"])
            merged["cfg_pe_max"]       = int(merged["cfg_pe_max"])
            merged["cfg_div_min"]      = float(merged["cfg_div_min"])
            merged["cfg_ret_w1_min"]   = float(merged["cfg_ret_w1_min"])
            merged["cfg_ret_m1_min"]   = float(merged["cfg_ret_m1_min"])
            return merged
        except Exception:
            pass
    return _DEFAULTS_CFG.copy()

def guardar_config():
    """Guarda la configuración actual (desde session_state) en disco."""
    try:
        data = {k: st.session_state[k] for k in _DEFAULTS_CFG if k in st.session_state}
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ──────────────────────────────────────────────
# EMBEDDINGS SEMÁNTICOS (OpenAI text-embedding-3-small)
# ──────────────────────────────────────────────
EMBS_PATH = BASE_DIR / "embs_openai.csv"

@st.cache_data
def cargar_embeddings():
    """Carga los embeddings guardados. Devuelve (DataFrame, n) o (None, 0)."""
    if not os.path.exists(EMBS_PATH):
        return None, 0
    df = pd.read_csv(EMBS_PATH, index_col=0)
    return df, len(df)

def generar_embeddings_openai(api_key: str, progress_bar, status_text) -> tuple[bool, str]:
    """
    Lee longBusinessSummary de bbdd_yahoo.csv, genera embeddings con OpenAI
    text-embedding-3-small y guarda el resultado en embs_openai.csv.
    """
    from openai import OpenAI
    YAHOO_PATH = BASE_DIR / "bbdd_yahoo.csv"
    try:
        yf = pd.read_csv(YAHOO_PATH, usecols=["ticker", "longBusinessSummary"])
    except Exception as e:
        return False, f"No se pudo leer bbdd_yahoo.csv: {e}"

    yf = yf.dropna(subset=["longBusinessSummary"]).copy()
    yf["ticker"] = yf["ticker"].str.upper()
    yf["text"]   = yf["longBusinessSummary"].str[:1200]   # ~300 tokens max

    tickers = yf["ticker"].tolist()
    texts   = yf["text"].tolist()
    total   = len(texts)

    client = OpenAI(api_key=api_key)
    BATCH  = 100
    rows   = []

    for i in range(0, total, BATCH):
        bt = tickers[i : i + BATCH]
        bx = texts[i : i + BATCH]
        status_text.text(f"⚙️ Procesando {i+1}–{min(i+BATCH, total)} de {total} empresas…")
        try:
            resp = client.embeddings.create(input=bx, model="text-embedding-3-small")
            for j, emb in enumerate(resp.data):
                rows.append([bt[j]] + emb.embedding)
        except Exception as e:
            return False, f"Error en lote {i//BATCH + 1}: {e}"
        progress_bar.progress(min((i + BATCH) / total, 1.0))

    cols   = ["ticker"] + list(range(1536))
    df_emb = pd.DataFrame(rows, columns=cols).set_index("ticker")
    df_emb.to_csv(EMBS_PATH)
    return True, f"✅ {len(df_emb):,} embeddings generados y guardados."

def buscar_similar(query: str, api_key: str, embs_df, top_n: int) -> list:
    """
    Convierte el query en un embedding y devuelve los top_n tickers
    más similares por producto punto (cosine similarity).
    """
    from openai import OpenAI
    client  = OpenAI(api_key=api_key)
    resp    = client.embeddings.create(input=[query], model="text-embedding-3-small")
    q_vec   = np.array(resp.data[0].embedding, dtype=float)
    matrix  = embs_df.values.astype(float)
    sims    = matrix.dot(q_vec)
    idx     = np.argsort(sims)[::-1][:top_n]
    return embs_df.index[idx].tolist()


df_full = cargar_datos()
sectores = sorted(df_full["SectorName"].dropna().unique().tolist())

# ──────────────────────────────────────────────
# FUNCIONES DE IA
# ──────────────────────────────────────────────
def construir_sistema(df_resultado: pd.DataFrame) -> str:
    """Genera el system prompt con contexto de los resultados actuales."""
    n = len(df_resultado)
    if n == 0:
        resumen = "No hay empresas que cumplan los filtros actuales."
    else:
        top5 = df_resultado.head(5)[["Ticker","Name","SectorName","puntaje"]].to_string(index=False)
        resumen = f"""Hay {n} empresas en los resultados actuales.
Puntaje promedio: {df_resultado['puntaje'].mean():.1f}/100
Top 5:
{top5}"""

    return f"""Eres un asistente financiero experto que ayuda a un inversor principiante
a entender los resultados de un screener de acciones basado en datos de Morningstar.

CONTEXTO DE LOS RESULTADOS ACTUALES:
{resumen}

BASE DE DATOS COMPLETA: {len(df_full):,} empresas de NYSE y Nasdaq.

MÉTRICAS DISPONIBLES Y SU SIGNIFICADO:
- puntaje (0-100): puntuación de calidad combinada
- QuantitativeStarRating: rating Morningstar de 1 a 5 estrellas
- ROEYear1: retorno sobre capital propio (>15% es bueno)
- ROICYear1: retorno sobre capital invertido (>10% es bueno)
- NetMargin: margen de ganancia neta (>10% es bueno)
- DebtEquityRatio: deuda/capital (menor es mejor, <1.5 es sano)
- EPSGrowth3YYear1: crecimiento de ganancias en 3 años
- RevenueGrowth3Y: crecimiento de ventas en 3 años (%)
- PERatio: precio/ganancias (valuación)
- DividendYield: rendimiento por dividendos (%)
- MarketCap: capitalización de mercado en USD

INSTRUCCIONES:
- Explicá los conceptos financieros en lenguaje simple y claro
- Cuando menciones porcentajes de las métricas, convertí decimales a %
  (por ej. ROE de 0.25 = 25%)
- Nunca des recomendaciones de compra/venta concretas
- Si te preguntan por una empresa específica, buscá información en el contexto
- Respondé siempre en español
"""

def stream_claude(messages_hist, system_prompt, api_key):
    """Genera respuesta en streaming con Claude."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages_hist,
    ) as stream:
        for text in stream.text_stream:
            yield text

def stream_openai(messages_hist, system_prompt, api_key):
    """Genera respuesta en streaming con OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    msgs = [{"role": "system", "content": system_prompt}] + messages_hist
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        stream=True,
        max_tokens=1024,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def obtener_info_empresa(ticker: str) -> str:
    """Devuelve un resumen de métricas de una empresa dada su ticker."""
    row = df_full[df_full["Ticker"].str.upper() == ticker.upper()]
    if row.empty:
        return f"No encontré información para el ticker '{ticker}'."
    r = row.iloc[0]
    return f"""
**{r.get('Name', ticker)} ({ticker})**
- Sector: {r.get('SectorName','N/A')} | Industria: {r.get('IndustryName','N/A')}
- Puntaje calidad: {r.get('puntaje', np.nan):.0f}/100
- Rating MS: {r.get('QuantitativeStarRating','N/A')} ⭐
- ROE: {r.get('ROEYear1', np.nan)*100:.1f}% | ROIC: {r.get('ROICYear1', np.nan)*100:.1f}%
- Margen neto: {r.get('NetMargin', np.nan)*100:.1f}%
- Deuda/Capital: {r.get('DebtEquityRatio','N/A'):.2f}
- Crec. EPS 3Y: {r.get('EPSGrowth3YYear1', np.nan)*100:.1f}%
- Crec. Ventas 3Y: {r.get('RevenueGrowth3Y','N/A'):.1f}%
- P/E: {r.get('PERatio','N/A')} | Dividendo: {r.get('DividendYield','N/A')}%
- Market Cap: ${r.get('MarketCap', 0)/1e9:.2f}B
- Precio: ${r.get('ClosePrice','N/A')}
"""

# ──────────────────────────────────────────────
# ESTADO DE SESIÓN
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cargar configuración guardada (solo la primera vez por sesión)
# IMPORTANTE: va ANTES de los inits individuales para que setdefault
# pueda aplicar los valores del JSON (sem_tickers, sem_query_prev, etc.)
if "_cfg_loaded" not in st.session_state:
    for _k, _v in cargar_config().items():
        st.session_state.setdefault(_k, _v)
    st.session_state["_cfg_loaded"] = True

# Fallbacks por si no estaban en el JSON guardado
if "sem_tickers" not in st.session_state:
    st.session_state.sem_tickers = None        # None = sin filtro semántico activo
if "sem_query_prev" not in st.session_state:
    st.session_state.sem_query_prev = ""

# Cargar embeddings una sola vez (cached)
_embs_df, _n_embs = cargar_embeddings()
_tiene_embs = _embs_df is not None
sem_activa  = False   # se actualiza en el sidebar

# ── CSS global: reposicionar tooltips del "?" nativo fuera del sidebar ──
st.markdown("""
<style>
/* Desplaza el panel contenedor del tooltip BaseUI hacia la derecha del sidebar */
[data-autofocus-inside]:has([role="tooltip"][data-baseweb]) {
    transform: translateX(358px) !important;
}
/* Texto más grande y caja más ancha dentro del tooltip */
[role="tooltip"][data-baseweb] {
    font-size: 14px !important;
    line-height: 1.7 !important;
    max-width: 380px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR: CONFIGURACIÓN Y FILTROS
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")

    st.subheader("🤖 Proveedor de IA")
    proveedor = st.radio(
        "Elegí el modelo:",
        ["Claude (Anthropic)", "GPT-4o (OpenAI)"],
        key="cfg_proveedor",
        help="Necesitás una API key del proveedor elegido.",
    )

    if proveedor == "Claude (Anthropic)":
        api_key = st.text_input(
            "API Key de Anthropic",
            type="password",
            placeholder="sk-ant-...",
            help="Conseguila gratis en console.anthropic.com",
        )
        modelo_label = "claude-opus-4-6"
    else:
        api_key = st.text_input(
            "API Key de OpenAI",
            type="password",
            placeholder="sk-...",
            help="Conseguila en platform.openai.com",
        )
        modelo_label = "gpt-4o"

    # Cachear la key de OpenAI en session_state para usarla en embeddings
    # incluso cuando el proveedor de chat esté en Claude
    if proveedor == "GPT-4o (OpenAI)" and api_key:
        st.session_state["_oai_key"] = api_key
    _oai_key = st.session_state.get("_oai_key", "")

    st.caption(f"Modelo: `{modelo_label}`")
    st.divider()

    # ── FILTROS ──
    st.subheader("📊 Filtros del Screener")

    puntaje_min = st.slider(
        "Puntaje mínimo de calidad",
        min_value=0, max_value=100, value=60, step=5,
        key="cfg_puntaje_min",
        help=(
            "Puntaje propio (0 a 100) que combina 7 métricas:\n\n"
            "• ⭐ Rating Morningstar ≥ 4  → 20 pts\n"
            "• 💰 ROE > 15%              → 15 pts\n"
            "• 🏭 ROIC > 10%             → 15 pts\n"
            "• 📈 Margen neto > 10%      → 15 pts\n"
            "• 🏦 Deuda/Capital < 1.5    → 15 pts\n"
            "• 📊 Crec. ganancias > 0%   → 10 pts\n"
            "• 📦 Crec. ventas > 5%      → 10 pts\n\n"
            "Recomendado: empezá en 60. "
            "Subilo para ver solo las más sólidas."
        ),
    )

    sector_sel = st.multiselect(
        "Sector (vacío = todos)",
        options=sectores,
        default=[],
        key="cfg_sector_sel",
        help=(
            "Filtrá por sector económico.\n\n"
            "Ejemplos:\n"
            "• Technology → empresas de software, chips, etc.\n"
            "• Healthcare → farmacéuticas y hospitales\n"
            "• Consumer Defensive → alimentos, bebidas, supermercados\n"
            "• Financials → bancos, aseguradoras\n"
            "• Industrials → manufactura, logística\n\n"
            "Tip: dejalo vacío para ver todos los sectores "
            "y luego filtrá por sector si encontrás algo interesante."
        ),
    )

    mcap_min = st.slider(
        "Market Cap mínimo (USD Billion)",
        min_value=0.0, max_value=50.0, value=1.0, step=0.5,
        key="cfg_mcap_min",
        help=(
            "Market Cap = precio de la acción × cantidad de acciones.\n"
            "Es el 'tamaño' de la empresa en dólares.\n\n"
            "Referencia:\n"
            "• < $2B   → empresa pequeña (más riesgo, más potencial)\n"
            "• $2–10B  → empresa mediana\n"
            "• > $10B  → empresa grande (más estable)\n"
            "• > $200B → gigantes (Apple, Microsoft, etc.)\n\n"
            "Tip: si sos principiante, quedáte en empresas "
            "grandes (> $10B). Son más predecibles."
        ),
    )

    pe_max = st.number_input(
        "P/E máximo (0 = sin límite)",
        min_value=0, max_value=500, value=40,
        key="cfg_pe_max",
        help=(
            "P/E (Price-to-Earnings) = precio / ganancias por acción.\n"
            "Indica cuántos años de ganancias estás pagando.\n\n"
            "Referencia:\n"
            "• P/E < 15  → barato (o empresa en problemas)\n"
            "• P/E 15–25 → valuación razonable\n"
            "• P/E 25–40 → caro, el mercado espera mucho crecimiento\n"
            "• P/E > 40  → muy caro o empresa de altísimo crecimiento\n\n"
            "Tip: ponelo en 0 para no filtrar por valuación "
            "y ver todas las empresas de calidad sin importar el precio."
        ),
    )

    div_min = st.number_input(
        "Dividendo mínimo % (0 = incluir sin dividendo)",
        min_value=0.0, max_value=20.0, value=0.0, step=0.5,
        key="cfg_div_min",
        help=(
            "Dividendo = parte de las ganancias que la empresa "
            "reparte en efectivo a sus accionistas.\n\n"
            "Referencia:\n"
            "• 0%     → no paga dividendo (reinvierte todo)\n"
            "• 1–2%   → dividendo bajo\n"
            "• 2–5%   → dividendo atractivo\n"
            "• > 5%   → dividendo alto (¡verificar que sea sostenible!)\n\n"
            "Tip: dejalo en 0 si no necesitás ingresos regulares. "
            "Las empresas de mayor crecimiento suelen no pagar dividendo."
        ),
    )

    st.divider()

    # ── SEÑALES DE CORTO PLAZO ──
    _tiene_cp = "ma50_pct" in df_full.columns
    st.subheader("⚡ Señales de Corto Plazo")

    if not _tiene_cp:
        st.caption("⚠️ bbdd_yahoo.csv no encontrado. Estos filtros no están disponibles.")
        ma50_on = ma200_on = vol_on = False
        ret_w1_min = ret_m1_min = 0.0
    else:
        st.caption("Combiná con los filtros de calidad para encontrar "
                   "empresas sólidas en buen momento técnico.")

        ma50_on = st.checkbox(
            "📈 Precio sobre Media 50 días",
            value=False,
            key="cfg_ma50_on",
            help=(
                "Filtra empresas cuyo precio actual está\n"
                "POR ENCIMA de su media móvil de 50 días.\n\n"
                "La MA50 refleja la tendencia de corto-mediano\n"
                "plazo. Precio sobre MA50 = el mercado está\n"
                "empujando la acción hacia arriba.\n\n"
                "Tip: combinalo con MA200 para doble confirmación\n"
                "de tendencia alcista."
            ),
        )

        ma200_on = st.checkbox(
            "📊 Precio sobre Media 200 días",
            value=False,
            key="cfg_ma200_on",
            help=(
                "Filtra empresas cuyo precio actual está\n"
                "POR ENCIMA de su media móvil de 200 días.\n\n"
                "La MA200 es la referencia técnica más seguida\n"
                "por fondos e instituciones globales.\n"
                "Cruzar hacia arriba = señal alcista de largo plazo.\n\n"
                "Tip: precio sobre MA50 Y MA200 → señal muy fuerte."
            ),
        )

        vol_on = st.checkbox(
            "🔊 Volumen inusual (> 1.5× el promedio)",
            value=False,
            key="cfg_vol_on",
            help=(
                "Filtra empresas con volumen mayor a 1.5 veces\n"
                "su promedio de los últimos 10 días.\n\n"
                "Volumen alto = dinero institucional entrando,\n"
                "o noticias relevantes atrayendo atención.\n"
                "Una suba con volumen alto es más confiable\n"
                "que una suba con volumen bajo.\n\n"
                "⚠️ El volumen del CSV depende de cuándo se\n"
                "generó bbdd_yahoo.csv."
            ),
        )

        ret_w1_min = st.slider(
            "Retorno mínimo última semana (%)",
            min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
            key="cfg_ret_w1_min",
            help=(
                "Filtra empresas con retorno semanal ≥ al valor.\n\n"
                "• Positivo → solo empresas subiendo esta semana\n"
                "• 0        → sin filtro por retorno semanal\n"
                "• Negativo → acepta también las que bajaron poco\n\n"
                "Tip: entre +1% y +3% captura momentum sin\n"
                "descartar demasiados candidatos."
            ),
        )

        ret_m1_min = st.slider(
            "Retorno mínimo último mes (%)",
            min_value=-30.0, max_value=30.0, value=0.0, step=1.0,
            key="cfg_ret_m1_min",
            help=(
                "Filtra empresas con retorno mensual ≥ al valor.\n\n"
                "Un retorno mensual positivo confirma que el mercado\n"
                "está apostando a esta empresa con dinero real.\n\n"
                "Tip: combiná filtro semanal + mensual para verificar\n"
                "que el momentum es sostenido y no un spike aislado."
            ),
        )

    st.divider()

    # ── BÚSQUEDA SEMÁNTICA ──
    st.subheader("🔍 Búsqueda Semántica")

    if not _tiene_embs:
        st.caption("Embeddings no generados aún. "
                   "Generalos en la sección **Actualizar datos** de abajo.")
        sem_activa = False
    else:
        _fecha_embs = datetime.fromtimestamp(os.path.getmtime(EMBS_PATH)).strftime("%d/%m/%Y")
        st.caption(f"📚 {_n_embs:,} empresas indexadas · actualizado {_fecha_embs}")

        sem_query = st.text_area(
            "Describí el tipo de empresa que buscás",
            value=st.session_state.sem_query_prev,
            placeholder=(
                "Ej: proveedoras de chips para IA, "
                "energía solar, farmacéuticas de oncología…"
            ),
            height=90,
            help=(
                "Escribí en español o inglés — el modelo entiende los dos.\n\n"
                "No hace falta usar palabras exactas. El sistema busca\n"
                "similitud de concepto, no de texto.\n\n"
                "Ejemplos:\n"
                "• 'empresas que se benefician del gasto en defensa'\n"
                "• 'compañías con modelos de suscripción recurrente'\n"
                "• 'fabricantes de equipos médicos de diagnóstico'\n"
                "• 'infraestructura para centros de datos e IA'\n\n"
                "Dejalo vacío para desactivar el filtro semántico."
            ),
        )

        sem_top_n = st.slider(
            "Empresas candidatas a considerar",
            min_value=20, max_value=500, value=150, step=10,
            help=(
                "Cuántas empresas tomar del ranking semántico antes\n"
                "de aplicar los filtros de calidad y corto plazo.\n\n"
                "• Menos (20–50)  → solo las más parecidas a tu idea\n"
                "• Más (200–500)  → red más amplia, más variedad\n\n"
                "Tip: empezá en 150 y ajustá según cuántos resultados\n"
                "finales aparecen en la tabla."
            ),
        )

        _col_b, _col_l = st.columns([2, 1])
        _buscar  = _col_b.button("🔍 Buscar",  use_container_width=True,
                                  type="primary", disabled=not sem_query.strip())
        _limpiar = _col_l.button("✕ Limpiar", use_container_width=True)

        if _limpiar:
            st.session_state.sem_tickers   = None
            st.session_state.sem_query_prev = ""
            st.rerun()

        if _buscar:
            if not _oai_key:
                st.warning("⚠️ Necesitás una API key de OpenAI. "
                           "Seleccioná GPT-4o e ingresala arriba.")
            else:
                with st.spinner("Buscando empresas similares…"):
                    try:
                        _tickers_sem = buscar_similar(
                            sem_query.strip(), _oai_key, _embs_df, sem_top_n
                        )
                        st.session_state.sem_tickers    = _tickers_sem
                        st.session_state.sem_query_prev = sem_query.strip()
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Error en búsqueda: {_e}")

        sem_activa = st.session_state.sem_tickers is not None
        if sem_activa:
            st.success(f"🎯 Filtro activo · {len(st.session_state.sem_tickers)} candidatas")
            _q_short = st.session_state.sem_query_prev
            st.caption(f"_{_q_short[:65]}{'…' if len(_q_short) > 65 else ''}_")

    st.divider()

    # ── ACTUALIZAR DATOS ──
    st.subheader("🔄 Actualizar datos")
    st.caption(f"Última actualización: **{fecha_actualizacion()}**")

    exchanges_disp = list(EXCHANGES.keys())
    exchanges_nombres = list(EXCHANGES.values())
    exchanges_sel_nombres = st.multiselect(
        "Mercados a descargar",
        options=exchanges_nombres,
        default=["NYSE", "Nasdaq"],
        key="cfg_exchanges_nombres",
        help=(
            "Elegí de qué bolsas querés descargar datos.\n\n"
            "📌 Mercados disponibles:\n"
            "• NYSE      → Bolsa de Nueva York. ~3.000 empresas grandes y tradicionales "
            "(Coca-Cola, JPMorgan, ExxonMobil…)\n"
            "• Nasdaq    → Bolsa tecnológica. ~3.000 empresas "
            "(Apple, Microsoft, Nvidia, Meta…)\n"
            "• AMEX      → Mercado de empresas pequeñas y ETFs. ~300 empresas\n"
            "• Toronto   → Bolsa de Canadá. Fuerte en minería y energía\n"
            "• London    → Bolsa de Londres. Empresas europeas grandes\n"
            "• Frankfurt → Bolsa alemana. Referente europeo\n"
            "• Paris     → Bolsa francesa (Euronext)\n"
            "• Amsterdam → Bolsa holandesa (Euronext)\n\n"
            "⏱️ Tiempos estimados:\n"
            "• NYSE + Nasdaq     → ~6.000 empresas, ~1-2 min\n"
            "• Agregar 1 europeo → +500 empresas, +20-30 seg\n"
            "• Todos los mercados → ~8.000 empresas, ~3-5 min\n\n"
            "💡 Recomendación: empezá con NYSE + Nasdaq. "
            "Son los mercados más líquidos y analizados del mundo."
        ),
    )
    exchanges_sel_codes = [
        code for code, nombre in EXCHANGES.items()
        if nombre in exchanges_sel_nombres
    ]

    if st.button("⬇️ Descargar datos frescos", use_container_width=True,
                 type="primary", disabled=not exchanges_sel_codes):
        pb = st.progress(0)
        st_txt = st.empty()
        ok, msg = actualizar_datos_morningstar(exchanges_sel_codes, pb, s_txt := st.empty())
        pb.empty()
        s_txt.empty()
        if ok:
            st.success(msg)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(msg)

    st.caption(
        "ℹ️ Descarga datos en tiempo real desde Morningstar. "
        "Hacé clic solo cuando quieras datos más frescos que los actuales. "
        "El proceso puede tardar 1–5 min según los mercados elegidos. "
        "La app se recargará automáticamente al terminar."
    )

    # ── Generar embeddings semánticos ──
    st.markdown("---")
    if st.button("⚡ Generar embeddings semánticos", use_container_width=True):
        if not _oai_key:
            st.warning("⚠️ Seleccioná GPT-4o e ingresá tu API key de OpenAI primero.")
        else:
            pb2 = st.progress(0)
            st2 = st.empty()
            ok2, msg2 = generar_embeddings_openai(_oai_key, pb2, st2)
            pb2.empty()
            st2.empty()
            if ok2:
                st.success(msg2)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg2)

    st.caption(
        "⚙️ Genera los embeddings semánticos a partir de las descripciones "
        "de empresas en bbdd_yahoo.csv. Solo necesitás hacerlo una vez "
        "(o cuando actualices bbdd_yahoo.csv). Requiere API key de OpenAI."
    )

    # Guardar configuración automáticamente en cada interacción
    guardar_config()

    st.divider()
    if st.button(
        "🗑️ Limpiar chat",
        use_container_width=True,
        help=(
            "Borra todo el historial de la conversación con el asistente IA.\n\n"
            "Útil cuando:\n"
            "• Cambiás de tema o de empresa a analizar\n"
            "• El chat se volvió muy largo y querés empezar de cero\n"
            "• Querés que la IA no recuerde contexto anterior\n\n"
            "⚠️ Esta acción no se puede deshacer, "
            "pero los filtros y datos no se ven afectados."
        ),
    ):
        st.session_state.chat_history = []
        st.rerun()

# ──────────────────────────────────────────────
# APLICAR FILTROS
# ──────────────────────────────────────────────
mask = df_full["puntaje"].notna()
mask &= df_full["puntaje"] >= puntaje_min

if sector_sel:
    mask &= df_full["SectorName"].isin(sector_sel)

if mcap_min > 0:
    mask &= df_full["MarketCap"] >= mcap_min * 1e9

if pe_max > 0:
    mask &= (df_full["PERatio"] <= pe_max) | df_full["PERatio"].isna()

if div_min > 0:
    mask &= df_full["DividendYield"] >= div_min

# ── Filtros de corto plazo (solo cuando hay datos Yahoo) ──
if _tiene_cp:
    if ma50_on:
        mask &= df_full["ma50_pct"] > 0
    if ma200_on:
        mask &= df_full["ma200_pct"] > 0
    if vol_on:
        mask &= df_full["vol_ratio"] > 1.5
    if ret_w1_min != 0:
        mask &= df_full["ReturnW1"] >= ret_w1_min
    if ret_m1_min != 0:
        mask &= df_full["ReturnM1"] >= ret_m1_min

# ── Filtro semántico (solo cuando hay búsqueda activa) ──
if sem_activa and st.session_state.sem_tickers:
    mask &= df_full["Ticker"].isin(st.session_state.sem_tickers)

df_filtrado = (
    df_full[mask]
    .sort_values("puntaje", ascending=False)
    .reset_index(drop=True)
)
df_filtrado.index += 1
df_filtrado.index.name = "Rank"

# ──────────────────────────────────────────────
# PANEL PRINCIPAL
# ──────────────────────────────────────────────
st.title("📊 Screener de Calidad + Asistente IA")
st.caption("Filtrá empresas sólidas y preguntale al asistente lo que quieras.")

# ── Métricas resumen (KPI cards con tooltip nativo al hover) ──
_n  = len(df_filtrado)
_pt = f"{df_filtrado['puntaje'].mean():.1f}"      if _n else "—"
_re = f"{df_filtrado['ROEYear1'].mean()*100:.1f}%" if _n else "—"
_mn = f"{df_filtrado['NetMargin'].mean()*100:.1f}%" if _n else "—"

st.markdown(f"""
<style>
.ms-kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:4px}}
.ms-kpi{{background:#1e2130;border-radius:10px;padding:16px 18px;
         border:1px solid #2d3250;position:relative;cursor:help;
         transition:border-color .2s}}
.ms-kpi:hover{{border-color:#4a90d9}}
.ms-kpi-lbl{{font-size:13px;color:#9aa5be;margin-bottom:6px;
              text-decoration:underline dotted rgba(154,165,190,.45);
              text-underline-offset:3px}}
.ms-kpi-val{{font-size:28px;font-weight:700;color:#fff}}
.ms-kpi-tip{{
  visibility:hidden;opacity:0;
  position:absolute;top:50%;left:105%;transform:translateY(-50%);
  width:260px;background:#1a2040;color:#dce1f5;
  border:1px solid #4a90d9;border-radius:8px;
  padding:11px 14px;font-size:12px;line-height:1.7;
  z-index:9999;box-shadow:0 6px 22px rgba(0,0,0,.7);
  pointer-events:none;transition:opacity .2s
}}
.ms-kpi:hover .ms-kpi-tip{{visibility:visible;opacity:1}}
</style>
<div class="ms-kpi-grid">
  <div class="ms-kpi">
    <div class="ms-kpi-tip">
      Cantidad de empresas que cumplen <b>todos</b> los filtros activos.<br><br>
      Bajá el puntaje mínimo o ampliá los demás filtros si querés ver más resultados.
    </div>
    <div class="ms-kpi-lbl">Empresas encontradas</div>
    <div class="ms-kpi-val">{_n:,}</div>
  </div>
  <div class="ms-kpi">
    <div class="ms-kpi-tip">
      Puntaje promedio de calidad (escala 0–100) del grupo mostrado.<br><br>
      Promedio &gt; 70 → selección sólida<br>
      El puntaje suma: Rating MS, ROE, ROIC,<br>Margen, Deuda, Crec. EPS y Ventas.
    </div>
    <div class="ms-kpi-lbl">Puntaje promedio</div>
    <div class="ms-kpi-val">{_pt}</div>
  </div>
  <div class="ms-kpi">
    <div class="ms-kpi-tip">
      <b>ROE</b> = Return on Equity<br>
      (Retorno sobre el Patrimonio)<br><br>
      Cuánto gana la empresa por cada<br>dólar que pusieron los accionistas.<br><br>
      &lt; 10% → bajo<br>
      10–20% → aceptable<br>
      &gt; 20% → excelente<br>
      &gt; 30% → excepcional (Apple, MSFT…)
    </div>
    <div class="ms-kpi-lbl">ROE promedio</div>
    <div class="ms-kpi-val">{_re}</div>
  </div>
  <div class="ms-kpi">
    <div class="ms-kpi-tip">
      <b>Margen Neto</b> = Ganancia Neta / Ingresos<br><br>
      Cuánto queda de cada dólar vendido<br>después de todos los gastos.<br><br>
      &lt; 5%  → bajo (retail, supermercados)<br>
      5–15% → saludable<br>
      &gt; 20% → premium (software, farma)<br>
      &gt; 30% → excepcional
    </div>
    <div class="ms-kpi-lbl">Margen neto promedio</div>
    <div class="ms-kpi-val">{_mn}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabla de resultados ──
tab_calidad, tab_señales = st.tabs(["🏆 Calidad (largo plazo)", "⚡ Señales de entrada (corto plazo)"])

# ── Helper de formato para señales ──
def _fmt_pct(val):
    if pd.isna(val): return "—"
    v = float(val)
    return f"{'+' if v > 0 else ''}{v:.1f}%"

def _fmt_ratio(val):
    if pd.isna(val): return "—"
    return f"{float(val):.1f}×"

with tab_calidad:
  if df_filtrado.empty:
    st.warning("No hay empresas que cumplan los filtros. Probá relajando algún criterio.")
  else:
    # Columnas limpias para mostrar
    cols_show = ["Ticker","Name","SectorName","puntaje",
                 "QuantitativeStarRating","ROEYear1","ROICYear1",
                 "NetMargin","DebtEquityRatio","RevenueGrowth3Y",
                 "PERatio","DividendYield","MarketCap"]
    df_show = df_filtrado[cols_show].copy()

    # Formatear para mostrar
    df_show["ROEYear1"]       = (df_show["ROEYear1"] * 100).round(1).astype(str) + "%"
    df_show["ROICYear1"]      = (df_show["ROICYear1"] * 100).round(1).astype(str) + "%"
    df_show["NetMargin"]      = (df_show["NetMargin"] * 100).round(1).astype(str) + "%"
    df_show["RevenueGrowth3Y"]= df_show["RevenueGrowth3Y"].round(1).astype(str) + "%"
    df_show["MarketCap"]      = (df_show["MarketCap"] / 1e9).round(2).astype(str) + "B"
    df_show["puntaje"]        = df_show["puntaje"].round(0).astype(int)

    df_show.columns = ["Ticker","Empresa","Sector","Puntaje /100",
                       "Estrellas MS","ROE","ROIC","Mg Neto",
                       "Deuda/Cap","Crec.Ventas","P/E","Dividendo %","Mkt Cap"]

    # ── Leyenda interactiva: pasá el mouse sobre cada columna para ver qué significa ──
    _COL_TIPS = {
        "Rank":        ("Posición en el ranking, ordenada por puntaje de calidad.<br>"
                        "Rank 1 = empresa con mayor puntaje en el conjunto filtrado."),
        "Ticker":      ("Símbolo bursátil — código único de la empresa en bolsa.<br><br>"
                        "Ejemplos:<br>"
                        "AAPL = Apple &nbsp;|&nbsp; MSFT = Microsoft<br>"
                        "AMZN = Amazon &nbsp;|&nbsp; NVDA = Nvidia"),
        "Empresa":     "Nombre completo de la empresa listada en bolsa.",
        "Sector":      ("Sector económico al que pertenece la empresa.<br><br>"
                        "Tip: diversificá tu cartera eligiendo<br>"
                        "empresas de distintos sectores."),
        "Puntaje /100":("Puntaje de calidad propio (0–100).<br>"
                        "Combina 7 métricas:<br>"
                        "Rating MS · ROE · ROIC · Margen<br>"
                        "Deuda · Crec. EPS · Crec. Ventas"),
        "Estrellas MS":("Rating Morningstar (1–5 ⭐).<br><br>"
                        "4–5 ⭐ = empresa infravalorada según<br>"
                        "el modelo de valor intrínseco de Morningstar.<br>"
                        "1–2 ⭐ = cara según ese modelo."),
        "ROE":         ("Return on Equity = Ganancia / Patrimonio.<br><br>"
                        "Qué tan bien usa el dinero de sus accionistas.<br>"
                        "&lt; 10% → bajo<br>"
                        "10–20% → aceptable<br>"
                        "&gt; 20% → excelente"),
        "ROIC":        ("Return on Invested Capital.<br>"
                        "Similar al ROE pero incluye deuda → más robusto.<br><br>"
                        "&gt; 10% = la empresa crea valor real<br>"
                        "&gt; 15% = muy bueno<br>"
                        "&gt; 20% = excepcional"),
        "Mg Neto":     ("Margen Neto = Ganancia Neta / Ingresos.<br>"
                        "Cuánto queda por cada dólar vendido.<br><br>"
                        "&lt; 5%  → bajo (retail)<br>"
                        "5–15% → saludable<br>"
                        "&gt; 20% → premium (software, farma)"),
        "Deuda/Cap":   ("Deuda Total / Patrimonio Neto.<br>"
                        "Mide el apalancamiento de la empresa.<br><br>"
                        "&lt; 0.5 → muy conservadora<br>"
                        "0.5–1.5 → normal<br>"
                        "&gt; 1.5 → riesgo elevado en crisis"),
        "Crec.Ventas": ("Crecimiento de ingresos promedio<br>"
                        "en los últimos 3 años.<br><br>"
                        "&lt; 0%  → empresa estancada o en declive<br>"
                        "0–5%  → crecimiento lento<br>"
                        "&gt; 5%  → en expansión ✅<br>"
                        "&gt; 20% → hipercrecimiento"),
        "P/E":         ("Price-to-Earnings = Precio / Ganancia por acción.<br>"
                        "Cuántos años de ganancias estás pagando.<br><br>"
                        "&lt; 15 → barato (o en problemas)<br>"
                        "15–25 → razonable<br>"
                        "25–40 → el mercado espera crecimiento<br>"
                        "&gt; 40 → muy caro"),
        "Dividendo %": ("Rendimiento por dividendo anual.<br>"
                        "% del precio de la acción que la empresa<br>"
                        "paga en efectivo a sus accionistas.<br><br>"
                        "0%   → reinvierte todo (más crecimiento)<br>"
                        "2–5% → dividendo atractivo<br>"
                        "&gt;5% → verificar que sea sostenible"),
        "Mkt Cap":     ("Market Capitalization en miles de millones (B) de USD.<br>"
                        "Es el 'tamaño' de la empresa en bolsa.<br><br>"
                        "&lt; $2B  → pequeña (más riesgo/potencial)<br>"
                        "$2–10B  → mediana<br>"
                        "&gt; $10B → grande y estable<br>"
                        "&gt; $200B → gigantes globales"),
    }
    _pills_items = "".join(
        f'<span class="ms-col-pill">{col}'
        f'<span class="ms-col-tip">{tip}</span>'
        f'</span>'
        for col, tip in _COL_TIPS.items()
    )
    st.markdown(f"""
<style>
.ms-col-legend{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;align-items:center}}
.ms-col-legend-lbl{{font-size:11px;color:#6b7a9b;margin-right:4px;white-space:nowrap}}
.ms-col-pill{{
  display:inline-block;position:relative;
  background:#1e2130;border:1px solid #2d3250;border-radius:20px;
  padding:3px 11px;font-size:11.5px;color:#9aa5be;cursor:help;
  transition:all .15s
}}
.ms-col-pill:hover{{background:#253050;border-color:#4a90d9;color:#dce1f5}}
.ms-col-tip{{
  visibility:hidden;opacity:0;
  position:absolute;top:50%;left:105%;transform:translateY(-50%);
  width:230px;background:#1a2040;color:#dce1f5;
  border:1px solid #4a90d9;border-radius:8px;
  padding:10px 13px;font-size:11.5px;line-height:1.7;
  z-index:9999;box-shadow:0 6px 22px rgba(0,0,0,.75);
  pointer-events:none;transition:opacity .2s;white-space:normal
}}
.ms-col-pill:hover .ms-col-tip{{visibility:visible;opacity:1}}
</style>
<div class="ms-col-legend">
  <span class="ms-col-legend-lbl">🔍 Pasá el mouse sobre las columnas:</span>
  {_pills_items}
</div>
""", unsafe_allow_html=True)

    st.dataframe(
        df_show,
        use_container_width=True,
        height=300,
    )

with tab_señales:
    if not _tiene_cp:
        st.info(
            "Para ver señales de corto plazo necesitás tener **bbdd_yahoo.csv** "
            "en la misma carpeta. Ese archivo se genera en los notebooks de la clase.",
            icon="📊",
        )
    elif df_filtrado.empty:
        st.warning("No hay empresas que cumplan los filtros. Probá relajando algún criterio.")
    else:
        st.caption(
            "📌 Empresas que pasaron el filtro de calidad, vistas desde el ángulo técnico. "
            "Pasá el mouse sobre cada columna para ver qué significa."
        )

        # Columnas a mostrar (solo las que existen en el df)
        _cp_base  = ["Ticker","Name","SectorName","puntaje"]
        _cp_yahoo = ["ma50_pct","ma200_pct","vol_ratio","dist_52w_low","dist_52w_high","beta"]
        _cp_ms    = ["ReturnW1","ReturnM1","ReturnM3","ReturnD1"]
        _cols_cp  = _cp_base + [c for c in _cp_yahoo + _cp_ms if c in df_filtrado.columns]

        df_cp = df_filtrado[_cols_cp].copy()

        # Formatear
        for col in ["ma50_pct","ma200_pct","dist_52w_low","dist_52w_high",
                    "ReturnW1","ReturnM1","ReturnM3","ReturnD1"]:
            if col in df_cp.columns:
                df_cp[col] = df_cp[col].apply(_fmt_pct)
        if "vol_ratio" in df_cp.columns:
            df_cp["vol_ratio"] = df_cp["vol_ratio"].apply(_fmt_ratio)
        if "beta" in df_cp.columns:
            df_cp["beta"] = df_cp["beta"].round(2)
        df_cp["puntaje"] = df_cp["puntaje"].round(0).astype("Int64")

        df_cp = df_cp.rename(columns={
            "Name": "Empresa", "SectorName": "Sector", "puntaje": "Puntaje",
            "ma50_pct": "vs MA50", "ma200_pct": "vs MA200",
            "vol_ratio": "Vol.Ratio",
            "dist_52w_low": "↑ 52w-Low", "dist_52w_high": "↓ 52w-High",
            "ReturnW1": "Ret.1S", "ReturnM1": "Ret.1M",
            "ReturnM3": "Ret.3M",  "ReturnD1": "Ret.1D",
            "beta": "Beta",
        })

        # Leyenda con tooltips para las columnas de señales
        _CP_TIPS = {
            "Ticker":     "Símbolo bursátil. Usalo para buscar la empresa en cualquier bróker.",
            "Empresa":    "Nombre completo de la empresa.",
            "Sector":     "Sector económico.",
            "Puntaje":    "Puntaje de calidad (0–100). Solo aparecen empresas con buena salud financiera.",
            "vs MA50":    ("% que el precio está sobre (+) o bajo (–) la Media Móvil de 50 días.<br><br>"
                           "Positivo → tendencia alcista de corto plazo<br>"
                           "Negativo → debajo de la media, posible debilidad<br><br>"
                           "Tip: buscar empresas que cruzaron recientemente de negativo a positivo."),
            "vs MA200":   ("% sobre/bajo la Media Móvil de 200 días.<br>"
                           "Es la referencia técnica más seguida por instituciones.<br><br>"
                           "Positivo → tendencia de largo plazo confirmada ✅<br>"
                           "Negativo → zona de precaución ⚠️<br><br>"
                           "vs MA50 Y vs MA200 positivos → doble confirmación alcista."),
            "Vol.Ratio":  ("Volumen actual vs promedio de los últimos 10 días.<br><br>"
                           "1.0× → volumen normal<br>"
                           "&gt;1.5× → actividad inusual (posibles noticias)<br>"
                           "&gt;3.0× → movimiento muy relevante, revisar noticias"),
            "↑ 52w-Low":  ("% que el precio actual está por encima de su mínimo anual.<br><br>"
                           "0–20% → cerca del piso, zona de posible rebote<br>"
                           "20–60% → rango medio<br>"
                           "&gt;100% → muy por encima del mínimo (tendencia fuerte)"),
            "↓ 52w-High": ("% que el precio está por debajo de su máximo anual (siempre negativo).<br><br>"
                           "0% → precio EN el máximo anual (posible ruptura alcista)<br>"
                           "-10% a 0% → cerca del máximo, zona de resistencia<br>"
                           "&lt;-30% → lejos del máximo, puede estar en corrección"),
            "Ret.1S":     ("Retorno de la última semana (Morningstar).<br><br>"
                           "Positivo → momentum reciente favorable.<br>"
                           "Combinalo con Ret.1M para ver si es tendencia o spike."),
            "Ret.1M":     ("Retorno del último mes (Morningstar).<br><br>"
                           "Indica si el mercado está apostando a esta empresa<br>"
                           "con dinero real en el período reciente."),
            "Ret.3M":     ("Retorno de los últimos 3 meses (Morningstar).<br><br>"
                           "Confirma que el momentum no es solo un spike de un día<br>"
                           "sino una tendencia sostenida."),
            "Ret.1D":     "Retorno del último día (Morningstar). Cambio más reciente registrado.",
            "Beta":       ("Volatilidad relativa al mercado (S&P500).<br><br>"
                           "1.0 → se mueve igual que el mercado<br>"
                           "&gt;1 → más volátil (más riesgo y oportunidad en corto plazo)<br>"
                           "&lt;1 → más estable, movimientos más suaves"),
        }
        _pills_cp = "".join(
            f'<span class="ms-col-pill">{col}<span class="ms-col-tip">{tip}</span></span>'
            for col, tip in _CP_TIPS.items()
            if col in df_cp.columns
        )
        st.markdown(
            f'<div class="ms-col-legend">'
            f'<span class="ms-col-legend-lbl">🔍 Pasá el mouse sobre las columnas:</span>'
            f'{_pills_cp}</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(df_cp, use_container_width=True, height=350)

# ──────────────────────────────────────────────
# CHAT
# ──────────────────────────────────────────────
st.divider()
st.subheader("💬 Asistente IA – Preguntá lo que quieras")

if not api_key:
    st.info(
        f"👈 Ingresá tu API key de **{'Anthropic' if 'Claude' in proveedor else 'OpenAI'}** "
        "en el panel izquierdo para activar el chat.",
        icon="🔑",
    )
else:
    # Mostrar historial
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Sugerencias rápidas (solo si no hay historial)
    if not st.session_state.chat_history:
        st.caption("Sugerencias:")
        sugs = [
            "¿Qué significa el ROE y por qué importa?",
            "¿Cuáles son los mejores 3 resultados y por qué?",
            "¿Qué riesgo tiene una empresa con mucha deuda?",
            "Explicame el P/E en términos simples",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(sugs):
            if cols[i % 2].button(sug, use_container_width=True, key=f"sug_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                st.rerun()

    # Input del usuario
    user_input = st.chat_input("Escribí tu pregunta aquí…")

    if user_input:
        # Detectar consulta por ticker (ej: "qué es AAPL?", "MSFT", "dime sobre NVDA")
        palabras = user_input.upper().split()
        tickers_en_msg = [
            p.strip("¿?.,!") for p in palabras
            if p.strip("¿?.,!") in df_full["Ticker"].values
        ]

        msg_enriquecido = user_input
        if tickers_en_msg:
            infos = [obtener_info_empresa(t) for t in tickers_en_msg]
            msg_enriquecido += "\n\n[DATOS DE LA EMPRESA]:\n" + "\n".join(infos)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Historial para la API (usamos el mensaje enriquecido en el último turno)
        api_messages = []
        for m in st.session_state.chat_history[:-1]:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": msg_enriquecido})

        system_prompt = construir_sistema(df_filtrado)

        with st.chat_message("assistant"):
            try:
                if "Claude" in proveedor:
                    respuesta = st.write_stream(
                        stream_claude(api_messages, system_prompt, api_key)
                    )
                else:
                    respuesta = st.write_stream(
                        stream_openai(api_messages, system_prompt, api_key)
                    )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": respuesta}
                )

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "auth" in error_msg.lower() or "401" in error_msg:
                    st.error("❌ API key inválida. Verificá que la ingresaste correctamente.")
                elif "rate" in error_msg.lower() or "429" in error_msg:
                    st.error("⏳ Límite de velocidad alcanzado. Esperá unos segundos y volvé a intentar.")
                else:
                    st.error(f"❌ Error: {error_msg}")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "📊 Datos: Morningstar · "
    "🤖 IA: Anthropic Claude / OpenAI GPT-4o · "
    "⚠️ Solo informativo, no es recomendación de inversión."
)
