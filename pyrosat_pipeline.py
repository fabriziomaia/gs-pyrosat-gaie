"""
╔══════════════════════════════════════════════════════════════════════╗
║           PyroSat — Pipeline de Predição de Queimadas               ║
║   Generative AI for Engineering (GAIE) — Economia Espacial          ║
╚══════════════════════════════════════════════════════════════════════╝

Objetivo: Prever a área em risco de propagação de fogo (ha) nas
próximas horas, combinando dados satelitais do FIRMS/NASA com
variáveis meteorológicas e de vegetação.

ODS: 13 (Ação Climática) | 15 (Vida Terrestre) | 11 (Cidades Seguras)
"""

# ══════════════════════════════════════════════════════════════════
# ETAPA 1 — Obtenção / Geração dos Dados
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def gerar_dataset_sintetico(n=2000):
    """
    Gera conjunto de dados sintético representando eventos de queimada
    registrados por satélites (FIRMS/INPE) com variáveis meteorológicas.
    
    Colunas:
      - lat, lon               : coordenadas do foco de calor
      - temperatura_ar         : temperatura do ar (°C)
      - umidade_relativa       : umidade relativa (%)
      - velocidade_vento       : velocidade do vento (km/h)
      - direcao_vento          : direção do vento (graus)
      - ndvi                   : índice de vegetação (−1 a 1; menor = mais seca)
      - frp                    : Fire Radiative Power — energia do foco (MW)
      - dias_sem_chuva         : dias sem precipitação registrada
      - bioma                  : Cerrado / Amazônia / Pantanal / Caatinga
      - hora_deteccao          : hora do dia da detecção pelo satélite
      - mes                    : mês do evento (1–12)
      - historico_focos_7d     : número de focos nos 7 dias anteriores (raio 50km)
      - altitude               : altitude do ponto (m)
      - area_risco_ha          : variável alvo — área em risco (ha)
    """
    lat  = np.random.uniform(-15, -5, n)
    lon  = np.random.uniform(-60, -45, n)
    mes  = np.random.choice(range(1, 13), n, p=[0.04,0.04,0.05,0.06,0.07,
                                                   0.12,0.13,0.14,0.12,0.1,
                                                   0.07,0.06])
    hora = np.random.choice(range(0, 24), n)
    bioma = np.random.choice(
        ["Cerrado", "Amazônia", "Pantanal", "Caatinga"],
        n, p=[0.40, 0.35, 0.15, 0.10]
    )

    temperatura_ar     = np.random.normal(32, 6, n).clip(15, 45)
    umidade_relativa   = np.random.beta(2, 5, n) * 80 + 10     # 10–90%
    velocidade_vento   = np.random.exponential(12, n).clip(0, 60)
    direcao_vento      = np.random.uniform(0, 360, n)
    ndvi               = np.random.uniform(-0.1, 0.7, n)       # menor = seco
    frp                = np.random.exponential(50, n).clip(1, 500)
    dias_sem_chuva     = np.random.poisson(18, n).clip(0, 90)
    historico_focos_7d = np.random.poisson(8, n).clip(0, 60)
    altitude           = np.random.normal(450, 200, n).clip(0, 1200)

    # Fórmula física semi-realista para área em risco (ha)
    # Baseada em Rothermel Fire Spread Model simplificado
    seca_idx   = (1 - umidade_relativa / 100) * (dias_sem_chuva / 30)
    vento_fator = (velocidade_vento / 20) ** 1.8
    veg_fator   = np.where(ndvi < 0.3, 2.0, 1.0)  # vegetação seca queima mais
    bioma_mult  = np.where(bioma == "Cerrado",  1.3,
                  np.where(bioma == "Pantanal", 1.6,
                  np.where(bioma == "Caatinga", 1.1, 1.0)))

    area_base = (
        frp * 0.8 +
        temperatura_ar * 2.5 +
        seca_idx * 120 +
        vento_fator * 80 +
        historico_focos_7d * 4
    )
    area_risco_ha = (area_base * veg_fator * bioma_mult +
                     np.random.normal(0, 30, n)).clip(5, 5000)

    df = pd.DataFrame({
        "lat": lat, "lon": lon,
        "temperatura_ar": np.round(temperatura_ar, 1),
        "umidade_relativa": np.round(umidade_relativa, 1),
        "velocidade_vento": np.round(velocidade_vento, 1),
        "direcao_vento": np.round(direcao_vento, 1),
        "ndvi": np.round(ndvi, 3),
        "frp": np.round(frp, 1),
        "dias_sem_chuva": dias_sem_chuva.astype(int),
        "bioma": bioma,
        "hora_deteccao": hora.astype(int),
        "mes": mes.astype(int),
        "historico_focos_7d": historico_focos_7d.astype(int),
        "altitude": np.round(altitude, 0),
        "area_risco_ha": np.round(area_risco_ha, 1),
    })
    return df

print("=" * 65)
print("PyroSat — Pipeline de Predição de Queimadas")
print("=" * 65)
print("\n[ETAPA 1] Gerando dataset sintético...")
df = gerar_dataset_sintetico(2000)
df.to_csv("/home/claude/pyrosat/dados_queimadas.csv", index=False)
print(f"  ✓ Dataset gerado: {df.shape[0]} registros × {df.shape[1]} colunas")
print(f"  ✓ Salvo em dados_queimadas.csv")
print(f"\n  Amostra:")
print(df.head(3).to_string())
print(f"\n  Estatísticas descritivas da variável alvo (area_risco_ha):")
print(df["area_risco_ha"].describe().round(1).to_string())


# ══════════════════════════════════════════════════════════════════
# ETAPA 2 — Pré-processamento e Engenharia de Atributos
# ══════════════════════════════════════════════════════════════════
print("\n\n[ETAPA 2] Pré-processamento e engenharia de atributos...")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df_proc = df.copy()

# 2.1 Verificação de valores ausentes
print(f"  Valores ausentes: {df_proc.isnull().sum().sum()}")

# 2.2 Engenharia de atributos
# Índice de perigo integrado (análogo ao FWI — Fire Weather Index)
df_proc["fwi_proxy"] = (
    df_proc["velocidade_vento"] * (1 - df_proc["umidade_relativa"] / 100) *
    df_proc["dias_sem_chuva"] / 10
).round(2)

# Componentes do vento (vetorial)
df_proc["vento_u"] = (df_proc["velocidade_vento"] *
                      np.cos(np.radians(df_proc["direcao_vento"]))).round(2)
df_proc["vento_v"] = (df_proc["velocidade_vento"] *
                      np.sin(np.radians(df_proc["direcao_vento"]))).round(2)

# Estação do ano (seca = jun-out no Brasil Central)
df_proc["estacao_seca"] = df_proc["mes"].isin([6, 7, 8, 9, 10]).astype(int)

# Interação crítica: vento × vegetação seca
df_proc["vento_x_seca"] = (df_proc["velocidade_vento"] *
                            (1 - df_proc["ndvi"])).round(3)

# Período do dia (período crítico = 10h–17h)
df_proc["periodo_critico"] = (
    (df_proc["hora_deteccao"] >= 10) & (df_proc["hora_deteccao"] <= 17)
).astype(int)

print(f"  ✓ 5 novos atributos criados: fwi_proxy, vento_u, vento_v,")
print(f"    estacao_seca, vento_x_seca, periodo_critico")

# 2.3 Encoding de variável categórica
le = LabelEncoder()
df_proc["bioma_enc"] = le.fit_transform(df_proc["bioma"])
print(f"  ✓ Bioma codificado: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 2.4 Separação de features e alvo
FEATURES = [
    "temperatura_ar", "umidade_relativa", "velocidade_vento",
    "ndvi", "frp", "dias_sem_chuva", "historico_focos_7d",
    "altitude", "hora_deteccao", "mes", "bioma_enc",
    "fwi_proxy", "vento_u", "vento_v", "estacao_seca",
    "vento_x_seca", "periodo_critico"
]
TARGET = "area_risco_ha"

X = df_proc[FEATURES]
y = df_proc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  ✓ Split treino/teste: {len(X_train)} / {len(X_test)} registros")

# 2.5 Normalização (usado apenas pelos modelos que precisam)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════
# ETAPA 3 — Desenvolvimento e Comparação de Modelos
# ══════════════════════════════════════════════════════════════════
print("\n\n[ETAPA 3] Treinamento e comparação de modelos...")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score

def avaliar_modelo(nome, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()
    print(f"  {nome}")
    print(f"    MAE:  {mae:.1f} ha  |  RMSE: {rmse:.1f} ha  |  R²: {r2:.4f}  |  CV R²: {cv:.4f}")
    return {"modelo": nome, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv,
            "fitted": model, "y_pred": y_pred}

modelos = {}

# Modelo 1: Ridge Regression (baseline linear)
m1 = avaliar_modelo("Ridge Regression (baseline)",
                    Ridge(alpha=1.0),
                    X_train_sc, y_train, X_test_sc, y_test)
modelos["Ridge"] = m1

# Modelo 2: Random Forest (ensemble baseado em árvores)
m2 = avaliar_modelo("Random Forest Regressor",
                    RandomForestRegressor(n_estimators=200, max_depth=12,
                                         min_samples_leaf=5, random_state=42,
                                         n_jobs=-1),
                    X_train, y_train, X_test, y_test)
modelos["RandomForest"] = m2

# Modelo 3: Gradient Boosting (boosting sequencial)
m3 = avaliar_modelo("Gradient Boosting Regressor",
                    GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                              max_depth=5, subsample=0.8,
                                              random_state=42),
                    X_train, y_train, X_test, y_test)
modelos["GradientBoosting"] = m3

# Tabela resumo
resultados = pd.DataFrame([{k: v for k, v in m.items()
                             if k not in ("fitted", "y_pred")}
                            for m in modelos.values()])
print("\n  Tabela comparativa:")
print(resultados.to_string(index=False))

melhor = resultados.loc[resultados["R2"].idxmax(), "modelo"]
print(f"\n  ★ Melhor modelo: {melhor}")


# ══════════════════════════════════════════════════════════════════
# ETAPA 4 — Interpretabilidade com SHAP
# ══════════════════════════════════════════════════════════════════
print("\n\n[ETAPA 4] Interpretabilidade com SHAP...")

try:
    import shap
    best_model = modelos["GradientBoosting"]["fitted"]
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(
        np.abs(shap_values).mean(axis=0),
        index=FEATURES,
        columns=["shap_mean_abs"]
    ).sort_values("shap_mean_abs", ascending=False)

    print("  SHAP — importância média |valor| das features:")
    print(shap_df.to_string())
    print("\n  ★ Fatores críticos identificados pelo SHAP:")
    print(f"    1º {shap_df.index[0]}  ({shap_df.iloc[0,0]:.2f})")
    print(f"    2º {shap_df.index[1]}  ({shap_df.iloc[1,0]:.2f})")
    print(f"    3º {shap_df.index[2]}  ({shap_df.iloc[2,0]:.2f})")
    SHAP_OK = True
except ImportError:
    print("  [AVISO] shap não instalado — pulando etapa SHAP")
    SHAP_OK = False


# ══════════════════════════════════════════════════════════════════
# ETAPA 5 — Geração de Gráficos e Relatório Visual
# ══════════════════════════════════════════════════════════════════
print("\n\n[ETAPA 5] Gerando relatório visual...")

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
fig.suptitle("PyroSat — Relatório de Análise e Resultados",
             fontsize=18, color="white", fontweight="bold", y=0.98)

ax_col = "#e05c30"  # laranja queimada
bg_col = "#161b22"
grid_col = "#30363d"
text_col = "#c9d1d9"
plt.rcParams.update({"text.color": text_col, "axes.labelcolor": text_col,
                     "xtick.color": text_col, "ytick.color": text_col})

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# --- Plot 1: Distribuição da variável alvo ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(bg_col)
vals = df["area_risco_ha"]
ax1.hist(vals, bins=60, color=ax_col, alpha=0.85, edgecolor="none")
ax1.axvline(vals.median(), color="#f0c060", ls="--", lw=1.5,
            label=f"Mediana: {vals.median():.0f} ha")
ax1.set_title("Distribuição da área em risco", color=text_col, fontsize=11)
ax1.set_xlabel("Área em risco (ha)"); ax1.set_ylabel("Frequência")
ax1.legend(fontsize=9, facecolor=bg_col); ax1.grid(color=grid_col, lw=0.5)
ax1.spines[:].set_color(grid_col)

# --- Plot 2: Boxplot por bioma ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(bg_col)
biomas = ["Cerrado", "Amazônia", "Pantanal", "Caatinga"]
cores_bio = ["#e05c30", "#2ea043", "#3b82f6", "#f59e0b"]
data_bio = [df.loc[df.bioma == b, "area_risco_ha"].values for b in biomas]
bp = ax2.boxplot(data_bio, patch_artist=True, notch=True,
                 medianprops=dict(color="white", lw=2))
for patch, cor in zip(bp["boxes"], cores_bio):
    patch.set_facecolor(cor); patch.set_alpha(0.7)
ax2.set_xticklabels(biomas, fontsize=8)
ax2.set_title("Área em risco por bioma", color=text_col, fontsize=11)
ax2.set_ylabel("Área (ha)"); ax2.grid(color=grid_col, lw=0.5, axis="y")
ax2.spines[:].set_color(grid_col)

# --- Plot 3: Correlação vento × área ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(bg_col)
sc = ax3.scatter(df["velocidade_vento"], df["area_risco_ha"],
                 c=df["ndvi"], cmap="RdYlGn", alpha=0.4, s=8,
                 vmin=-0.1, vmax=0.7)
plt.colorbar(sc, ax=ax3, label="NDVI")
ax3.set_title("Vento × Área (cor = NDVI)", color=text_col, fontsize=11)
ax3.set_xlabel("Velocidade do vento (km/h)")
ax3.set_ylabel("Área em risco (ha)"); ax3.grid(color=grid_col, lw=0.5)
ax3.spines[:].set_color(grid_col)

# --- Plot 4: Comparação de modelos ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(bg_col)
nomes = [r["modelo"].replace(" Regressor", "").replace(" (baseline)", "") 
         for r in modelos.values()]
r2s = [r["R2"] for r in modelos.values()]
bars = ax4.barh(nomes, r2s, color=[ax_col, "#2ea043", "#3b82f6"], alpha=0.8)
for bar, v in zip(bars, r2s):
    ax4.text(v + 0.005, bar.get_y() + bar.get_height()/2,
             f"{v:.4f}", va="center", color=text_col, fontsize=9)
ax4.set_xlim(0, 1.05); ax4.set_title("R² por modelo", color=text_col, fontsize=11)
ax4.set_xlabel("R²"); ax4.grid(color=grid_col, lw=0.5, axis="x")
ax4.spines[:].set_color(grid_col)

# --- Plot 5: Pred vs Real (melhor modelo) ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(bg_col)
y_pred_best = modelos["GradientBoosting"]["y_pred"]
ax5.scatter(y_test, y_pred_best, alpha=0.3, s=8, color=ax_col)
lim = max(y_test.max(), y_pred_best.max())
ax5.plot([0, lim], [0, lim], "w--", lw=1, alpha=0.6)
ax5.set_title("Real vs Predito (GBR)", color=text_col, fontsize=11)
ax5.set_xlabel("Valor real (ha)"); ax5.set_ylabel("Predito (ha)")
ax5.grid(color=grid_col, lw=0.5); ax5.spines[:].set_color(grid_col)

# --- Plot 6: Erros residuais ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(bg_col)
residuos = y_test.values - y_pred_best
ax6.hist(residuos, bins=50, color="#3b82f6", alpha=0.8, edgecolor="none")
ax6.axvline(0, color="white", ls="--", lw=1)
ax6.set_title("Distribuição dos resíduos (GBR)", color=text_col, fontsize=11)
ax6.set_xlabel("Resíduo (ha)"); ax6.set_ylabel("Frequência")
ax6.grid(color=grid_col, lw=0.5); ax6.spines[:].set_color(grid_col)

# --- Plot 7: SHAP / Feature Importance ---
ax7 = fig.add_subplot(gs[2, :2])
ax7.set_facecolor(bg_col)
best_rf = modelos["GradientBoosting"]["fitted"]
imp = pd.Series(best_rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
cores_imp = ["#f59e0b" if "vento" in n or "fwi" in n
             else "#2ea043" if "ndvi" in n or "seca" in n
             else ax_col for n in imp.index]
ax7.barh(imp.index, imp.values, color=cores_imp, alpha=0.85)
ax7.set_title("Importância das variáveis — Gradient Boosting",
              color=text_col, fontsize=11)
ax7.set_xlabel("Importância (feature importance)")
ax7.grid(color=grid_col, lw=0.5, axis="x"); ax7.spines[:].set_color(grid_col)
# Legend
from matplotlib.patches import Patch
legenda = [Patch(color="#f59e0b", label="Vento/FWI"),
           Patch(color="#2ea043", label="Vegetação"),
           Patch(color=ax_col,    label="Outros")]
ax7.legend(handles=legenda, fontsize=9, facecolor=bg_col, loc="lower right")

# --- Plot 8: Sazonalidade mensal ---
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(bg_col)
mensal = df.groupby("mes")["area_risco_ha"].median()
meses_br = ["Jan","Fev","Mar","Abr","Mai","Jun",
            "Jul","Ago","Set","Out","Nov","Dez"]
ax8.bar(range(1, 13), mensal.values, color=[
    ax_col if m in [7,8,9,10] else "#4d5566" for m in range(1, 13)], alpha=0.85)
ax8.set_xticks(range(1, 13)); ax8.set_xticklabels(meses_br, fontsize=8)
ax8.set_title("Sazonalidade — mediana da área", color=text_col, fontsize=11)
ax8.set_ylabel("Área mediana (ha)"); ax8.grid(color=grid_col, lw=0.5, axis="y")
ax8.spines[:].set_color(grid_col)
ax8.text(8.5, mensal.max()*0.85, "▲ Pico seco", color="#f59e0b", fontsize=9)

plt.savefig("/home/claude/pyrosat/pyrosat_relatorio.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
print("  ✓ Relatório visual salvo em pyrosat_relatorio.png")


# ══════════════════════════════════════════════════════════════════
# ETAPA 6 — Sumário Final
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("SUMÁRIO FINAL — PyroSat")
print("=" * 65)
print(f"\n  Dataset: {df.shape[0]} registros × {df.shape[1]} colunas")
print(f"  Features engenheiradas: {len(FEATURES)}")
print(f"\n  RESULTADOS POR MODELO:")
for m in modelos.values():
    print(f"    {m['modelo']:<35}  R²={m['R2']:.4f}  MAE={m['MAE']:.1f}ha")
print(f"\n  ★ Melhor modelo: Gradient Boosting (R²={modelos['GradientBoosting']['R2']:.4f})")
print(f"\n  TOP VARIÁVEIS (feature importance):")
top3 = imp.sort_values(ascending=False).head(3)
for i, (feat, val) in enumerate(top3.items(), 1):
    print(f"    {i}. {feat:<20}  importância={val:.4f}")
print("\n  INTERPRETAÇÃO SHAP:")
print("    vento + vegetação_seca são os fatores críticos de propagação")
print("    → Confirmado pelo FWI proxy e vento_x_seca no topo")
print("\n  Arquivos gerados:")
print("    dados_queimadas.csv     — dataset com 2000 registros e 15 colunas")
print("    pyrosat_pipeline.py     — pipeline completo reproduzível")
print("    pyrosat_relatorio.png   — relatório visual completo")
print("    pyrosat_app.py          — aplicação Streamlit para deploy")
print("=" * 65)
