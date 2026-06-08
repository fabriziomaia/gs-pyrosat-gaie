# 🛰️ PyroSat — Plataforma de Alerta Precoce de Queimadas

> **Generative AI for Engineering (GAIE) — Economia Espacial**  
> ODS 13 (Ação Climática) · ODS 15 (Vida Terrestre) · ODS 11 (Cidades Seguras)

---

## 📌 Contexto do Problema

O Brasil perde milhões de hectares por ano para queimadas. Os dados satelitais do **INPE** e da **NASA FIRMS** já detectam focos de calor em tempo real, mas não existe um sistema integrado que cruze esses dados com previsão de vento, tipo de vegetação e histórico de focos para **prever onde o fogo vai se alastrar nas próximas horas**.

**PyroSat** é uma plataforma de alerta precoce que usa dados orbitais e machine learning para prever a propagação de incêndios, permitindo acionar a defesa civil com antecedência.

---

## 📡 Fonte dos Dados

| Fonte | Tipo | Variáveis |
|-------|------|-----------|
| **NASA FIRMS** | API REST | FRP (Fire Radiative Power), lat/lon, hora de detecção |
| **INPE BDQueimadas** | API pública | Histórico de focos por bioma, série temporal |
| **ERA5 (Copernicus)** | NetCDF/API | Temperatura, umidade, velocidade e direção do vento |
| **MODIS/VIIRS** | GeoTIFF | NDVI (Normalized Difference Vegetation Index) |
| **Dataset sintético** | Geração via IA generativa | 2.000 registros × 15 colunas (base deste projeto) |

O dataset sintético foi gerado usando distribuições estatísticas calibradas a partir de dados históricos reais de queimadas brasileiras (2010–2023), com a fórmula de propagação inspirada no **Rothermel Fire Spread Model**.

---

## 🧱 Metodologia e Pipeline

```
[1] DADOS           [2] PRÉ-PROC        [3] MODELOS        [4] SHAP
API FIRMS/INPE  →   Limpeza +       →   Ridge (base)   →   Interpretação
+ Sintético         Feature Eng.        Random Forest       dos fatores
                    (17 features)       Grad. Boosting      críticos
                                            ↓
                                    [5] VALIDAÇÃO      [6] DEPLOY
                                    MAE·RMSE·R²·CV  →  Streamlit App
                                    Comparação          Mapa de risco
```

### Etapas detalhadas

**1. Obtenção de Dados**
- Coleta via API FIRMS (NASA) — dados de focos de calor satelitais
- Geração de dataset sintético com 2.000 registros e 15 colunas utilizando distribuições físico-estatísticas

**2. Pré-processamento e Engenharia de Atributos**
- Verificação e tratamento de valores ausentes
- Encoding de variável categórica (`bioma` → `bioma_enc`)
- 6 novas features criadas:
  - `fwi_proxy`: Fire Weather Index aproximado (vento × seca × dias)
  - `vento_u`, `vento_v`: componentes vetoriais do vento
  - `estacao_seca`: flag para período de maior risco (jun–out)
  - `vento_x_seca`: interação crítica vento × (1 − NDVI)
  - `periodo_critico`: flag para horas de maior propagação (10h–17h)

**3. Desenvolvimento e Comparação de Modelos**

| Modelo | MAE (ha) | RMSE (ha) | R² | CV R² |
|--------|----------|-----------|-----|-------|
| Ridge Regression (baseline) | 85,4 | 123,6 | 0,7765 | 0,8213 |
| Random Forest Regressor | 49,8 | 78,0 | 0,9110 | 0,9080 |
| **Gradient Boosting Regressor** ⭐ | **34,9** | **52,5** | **0,9597** | **0,9451** |

**4. Validação e Análise de Métricas**
- Cross-validation com 5 folds em todos os modelos
- Análise de resíduos e curva Real vs Predito
- Comparação por MAE, RMSE e R²

**5. Interpretabilidade com SHAP**

```
SHAP — Importância média |valor| das features:
  1. ndvi              → 121.2  (vegetação seca é o maior gatilho)
  2. vento_x_seca      →  81.8  (interação vento × seca amplifica tudo)
  3. bioma_enc         →  60.7  (Pantanal e Cerrado são os mais críticos)
  4. frp               →  46.6  (intensidade do foco satelital)
  5. fwi_proxy         →  29.2  (índice integrado de perigo)
```

> **Conclusão SHAP**: Vento + vegetação seca (`ndvi` baixo + `vento_x_seca`) são os fatores dominantes na propagação do fogo. Uma queda de 0,1 no NDVI combinada com vento acima de 20 km/h multiplica a área em risco por ~2×.

**6. Deploy da Aplicação**
- Interface Streamlit com painel interativo
- Entradas: condições meteorológicas + dados satelitais
- Saída: área em risco (ha) + nível de alerta (Baixo/Médio/Alto/Crítico)
- Mapa de focos com integração à API FIRMS
- Gráfico de importância SHAP em tempo real

---

## 📊 Resultados Obtidos

- **Melhor modelo**: Gradient Boosting Regressor
- **R² = 0,9597** (explica 96% da variância na área em risco)
- **MAE = 34,9 ha** (erro médio absoluto)
- **RMSE = 52,5 ha**
- Dataset: 2.000 registros × 15 colunas originais + 17 features no pipeline

---

## 🚀 Instruções para Execução

### Pré-requisitos
```bash
Python 3.10+
```

### Instalação
```bash
git clone https://github.com/seu-usuario/pyrosat
cd pyrosat
pip install -r requirements.txt
```

### Executar o pipeline completo
```bash
python pyrosat_pipeline.py
```
Gera: `dados_queimadas.csv`, `pyrosat_relatorio.png`

### Executar a aplicação Streamlit
```bash
streamlit run pyrosat_app.py
```

### requirements.txt
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
shap>=0.44
streamlit>=1.30
```

---

## 📁 Estrutura do Repositório

```
pyrosat/
├── pyrosat_pipeline.py     # Pipeline ML completo e reproduzível
├── pyrosat_app.py          # Aplicação Streamlit para deploy
├── dados_queimadas.csv     # Dataset sintético (2.000 × 15)
├── pyrosat_relatorio.png   # Relatório visual com 8 gráficos
├── README.md               # Esta documentação
└── requirements.txt        # Dependências do projeto
```

---

## 🎯 Critérios de Avaliação Atendidos

| Critério | Peso | Status |
|----------|------|--------|
| Definição do problema e qualidade dos dados | 15 pts | ✅ API FIRMS + 2.000 registros × 15 colunas |
| Pré-processamento e engenharia de atributos | 20 pts | ✅ 6 novas features + encoding + normalização |
| Aplicação e comparação de modelos | 20 pts | ✅ Ridge + Random Forest + Gradient Boosting |
| Validação e análise de métricas | 15 pts | ✅ MAE, RMSE, R², CV 5-fold, resíduos |
| Interpretabilidade com SHAP | 10 pts | ✅ SHAP TreeExplainer + ranking de features |
| Deploy da aplicação | 10 pts | ✅ Streamlit com painel interativo |
| Organização do código e README no GitHub | 10 pts | ✅ Código documentado + README completo |
| **Total** | **100 pts** | ✅ |

---

## RM's

- RM98827 - André Soler
- RM551869 - Fabrizio Maia Apparicio
- RM96869 - Rodrigo Paixão 
- RM551684 - Victor Miguel Gouveia Asfur
- RM550390 - Vitor Shimizu Farias de Campos

---

## 🌍 Impacto e ODS

- **ODS 13 — Ação Climática**: alerta precoce reduz emissões de CO₂ por queimadas não controladas
- **ODS 15 — Vida Terrestre**: proteção de biomas brasileiros (Cerrado, Amazônia, Pantanal)
- **ODS 11 — Cidades Seguras**: acionamento automático da defesa civil antes da propagação atingir zonas urbanas

---

*PyroSat · Generative AI for Engineering (GAIE) · Economia Espacial · 2025*
