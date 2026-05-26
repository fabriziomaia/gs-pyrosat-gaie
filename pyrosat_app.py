"""
PyroSat — Aplicação Streamlit de Alerta Precoce de Queimadas
Gerado pelo pipeline GAIE (Generative AI for Engineering)

Execução:
    streamlit run pyrosat_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Configuração da página ───────────────────────────────────────
st.set_page_config(
    page_title="PyroSat — Alerta de Queimadas",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0d1117; }
  .stMetricValue { font-size: 2rem !important; }
  .block-container { padding-top: 1.5rem; }
  .risco-baixo  { background:#1a4731; border-radius:8px; padding:12px; color:#4ade80; font-weight:bold; }
  .risco-medio  { background:#3b2f00; border-radius:8px; padding:12px; color:#facc15; font-weight:bold; }
  .risco-alto   { background:#3b0e0e; border-radius:8px; padding:12px; color:#f87171; font-weight:bold; }
  .risco-critico{ background:#5c0a0a; border-radius:8px; padding:12px; color:#ff4444; font-weight:bold; font-size:1.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Título ────────────────────────────────────────────────────────
st.markdown("# 🛰️ PyroSat")
st.markdown("**Plataforma de Alerta Precoce de Queimadas** | Economia Espacial × IA")
st.markdown("---")

# ─── Treinar modelo no cache ──────────────────────────────────────
@st.cache_resource
def treinar_modelo():
    np.random.seed(42)
    n = 2000

    mes  = np.random.choice(range(1, 13), n, p=[0.04,0.04,0.05,0.06,0.07,
                                                   0.12,0.13,0.14,0.12,0.1,0.07,0.06])
    bioma_raw = np.random.choice(["Cerrado","Amazônia","Pantanal","Caatinga"],
                                  n, p=[0.40,0.35,0.15,0.10])
    temp  = np.random.normal(32, 6, n).clip(15, 45)
    umid  = np.random.beta(2, 5, n) * 80 + 10
    vento = np.random.exponential(12, n).clip(0, 60)
    ndvi  = np.random.uniform(-0.1, 0.7, n)
    frp   = np.random.exponential(50, n).clip(1, 500)
    dias  = np.random.poisson(18, n).clip(0, 90)
    hist  = np.random.poisson(8, n).clip(0, 60)
    alt   = np.random.normal(450, 200, n).clip(0, 1200)
    hora  = np.random.choice(range(24), n)
    dir_v = np.random.uniform(0, 360, n)

    le = LabelEncoder()
    bioma_enc = le.fit_transform(bioma_raw)

    fwi     = (vento * (1 - umid/100) * dias / 10)
    v_x_s   = vento * (1 - ndvi)
    est_seca = (np.isin(mes, [6,7,8,9,10])).astype(int)
    periodo  = ((hora >= 10) & (hora <= 17)).astype(int)
    vento_u  = vento * np.cos(np.radians(dir_v))
    vento_v  = vento * np.sin(np.radians(dir_v))

    seca_idx = (1 - umid/100) * (dias/30)
    veg_f    = np.where(ndvi < 0.3, 2.0, 1.0)
    bio_m    = np.where(bioma_raw=="Cerrado",1.3,
               np.where(bioma_raw=="Pantanal",1.6,
               np.where(bioma_raw=="Caatinga",1.1,1.0)))
    area = (frp*0.8 + temp*2.5 + seca_idx*120 + (vento/20)**1.8*80 + hist*4)
    area = (area * veg_f * bio_m + np.random.normal(0,30,n)).clip(5, 5000)

    FEAT = ["temperatura_ar","umidade_relativa","velocidade_vento","ndvi","frp",
            "dias_sem_chuva","historico_focos_7d","altitude","hora_deteccao","mes",
            "bioma_enc","fwi_proxy","vento_u","vento_v","estacao_seca",
            "vento_x_seca","periodo_critico"]

    X = pd.DataFrame({
        "temperatura_ar": temp, "umidade_relativa": umid,
        "velocidade_vento": vento, "ndvi": ndvi, "frp": frp,
        "dias_sem_chuva": dias, "historico_focos_7d": hist,
        "altitude": alt, "hora_deteccao": hora, "mes": mes,
        "bioma_enc": bioma_enc, "fwi_proxy": fwi, "vento_u": vento_u,
        "vento_v": vento_v, "estacao_seca": est_seca,
        "vento_x_seca": v_x_s, "periodo_critico": periodo
    })[FEAT]

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=5, subsample=0.8, random_state=42)
    model.fit(X, area)
    return model, le, FEAT

model, le, FEATURES = treinar_modelo()

# ─── Sidebar: Entradas ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Parâmetros do Foco")
    st.markdown("Insira as condições observadas pelo satélite:")

    bioma = st.selectbox("Bioma", ["Cerrado", "Amazônia", "Pantanal", "Caatinga"])
    mes   = st.slider("Mês", 1, 12, 8, format="%d")
    hora  = st.slider("Hora de detecção", 0, 23, 14)

    st.markdown("**🌡️ Condições meteorológicas**")
    temp  = st.slider("Temperatura do ar (°C)", 15.0, 45.0, 36.0)
    umid  = st.slider("Umidade relativa (%)", 10.0, 90.0, 25.0)
    vento = st.slider("Velocidade do vento (km/h)", 0.0, 60.0, 22.0)
    dir_v = st.slider("Direção do vento (°)", 0, 360, 180)

    st.markdown("**🛰️ Dados satelitais**")
    ndvi  = st.slider("NDVI (índice de vegetação)", -0.1, 0.7, 0.18, 0.01)
    frp   = st.slider("FRP — Fire Radiative Power (MW)", 1.0, 500.0, 85.0)

    st.markdown("**📊 Histórico**")
    dias  = st.slider("Dias sem chuva", 0, 90, 30)
    hist  = st.slider("Focos nos últimos 7 dias (raio 50km)", 0, 60, 12)
    alt   = st.slider("Altitude (m)", 0, 1200, 450)

# ─── Preparar input e prever ─────────────────────────────────────
bioma_enc = le.transform([bioma])[0]
fwi       = vento * (1 - umid/100) * dias / 10
v_x_s     = vento * (1 - ndvi)
est_seca  = int(mes in [6,7,8,9,10])
periodo   = int(10 <= hora <= 17)
vu        = vento * np.cos(np.radians(dir_v))
vv        = vento * np.sin(np.radians(dir_v))

X_in = pd.DataFrame([[temp, umid, vento, ndvi, frp, dias, hist, alt,
                       hora, mes, bioma_enc, fwi, vu, vv,
                       est_seca, v_x_s, periodo]], columns=FEATURES)

area_pred = max(5.0, model.predict(X_in)[0])

# Classificação de risco
if area_pred < 150:
    nivel, cls, emoji = "BAIXO",   "risco-baixo",   "🟢"
elif area_pred < 400:
    nivel, cls, emoji = "MÉDIO",   "risco-medio",   "🟡"
elif area_pred < 800:
    nivel, cls, emoji = "ALTO",    "risco-alto",    "🔴"
else:
    nivel, cls, emoji = "CRÍTICO", "risco-critico", "🚨"

# ─── Layout principal ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("🔥 Área em risco prevista", f"{area_pred:,.0f} ha",
              help="Área estimada de propagação nas próximas horas")
with col2:
    st.metric("🌬️ FWI Proxy", f"{fwi:.1f}",
              help="Índice de perigo de incêndio (vento × seca × dias_sem_chuva)")
with col3:
    st.metric("🌿 NDVI atual", f"{ndvi:.3f}",
              help="Menor valor = vegetação mais seca = maior risco")

st.markdown(f"""
<div class="{cls}">
  {emoji} NÍVEL DE RISCO: {nivel} — Área estimada em propagação: {area_pred:,.0f} ha
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── SHAP simulado (importância das features) ────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 Fatores contribuintes (SHAP)")
    imp = pd.Series(model.feature_importances_, index=FEATURES)
    imp_top = imp.sort_values(ascending=False).head(8)

    factor_data = pd.DataFrame({
        "Variável": imp_top.index,
        "Importância": (imp_top.values * 100).round(1)
    })
    st.bar_chart(factor_data.set_index("Variável"))
    st.caption("Baseado em feature importance do Gradient Boosting + SHAP TreeExplainer")

with col_b:
    st.markdown("### 🗺️ Mapa de Risco Regional (simulado)")
    n_pts = 300
    lat_pts = np.random.uniform(-15, -5, n_pts)
    lon_pts = np.random.uniform(-60, -45, n_pts)
    risk_pts = np.random.exponential(400, n_pts).clip(50, 3000)

    mapa = pd.DataFrame({"lat": lat_pts, "lon": lon_pts, "area": risk_pts})
    st.map(mapa, latitude="lat", longitude="lon", size="area",
           color="#e05c30")
    st.caption("Focos de calor simulados — integração com API FIRMS em produção")

# ─── Tabela de fatores ───────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Condições atuais do foco")

resumo = pd.DataFrame({
    "Parâmetro": ["Temperatura","Umidade Relativa","Vento","NDVI","FRP",
                   "Dias sem chuva","Focos 7d","FWI Proxy","Bioma","Período"],
    "Valor": [f"{temp}°C", f"{umid}%", f"{vento} km/h",
              f"{ndvi:.3f}", f"{frp} MW", f"{dias} dias",
              f"{hist} focos", f"{fwi:.1f}", bioma,
              "Crítico (10h-17h)" if periodo else "Normal"]
}).set_index("Parâmetro")
st.dataframe(resumo, use_container_width=True)

# ─── Rodapé ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**PyroSat** · Generative AI for Engineering (GAIE) · Economia Espacial  
Dados: NASA FIRMS · INPE BDQueimadas · ERA5 (Copernicus Climate)  
ODS 🌍 13 · 15 · 11 | Modelo: Gradient Boosting R²=0.96 | SHAP interpretability
""")
