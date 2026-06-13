# =============================================================================
# CO-IA — Curadoria de Origem para IA
# app.py — Dashboard principal (Streamlit)
# =============================================================================

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    COLOR_ALERTA,
    COLOR_HUMANO,
    COLOR_NEUTRO,
    COLOR_SINTETICO,
    COLLAPSE_RISK_THRESHOLDS,
    DEFAULT_THRESHOLD,
    PATENT_REFERENCES,
)
from modules.modulo1_filtro import build_default_engine
from modules.modulo2_monitor import analisar_corpus
from modules.modulo3_provenance import ReportBuilder, gerar_proveniencia

# =============================================================================
# Configuracao de pagina
# =============================================================================

st.set_page_config(
    page_title="CO-IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Esconde elementos padrao do Streamlit */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* Tipografia base */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Botao de sidebar quando FECHADA — bem destacado */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: #1e4a6e !important;
        border-radius: 0 10px 10px 0 !important;
        padding: 12px 10px !important;
        margin-top: 20px !important;
        box-shadow: 3px 0 12px rgba(0,0,0,0.5) !important;
        z-index: 9999 !important;
        border: 1px solid #2a6a9a !important;
        border-left: none !important;
    }
    [data-testid="collapsedControl"] svg {
        color: #7ab8e0 !important;
        fill: #7ab8e0 !important;
        width: 20px !important;
        height: 20px !important;
    }
    /* Botao de fechar dentro da sidebar */
    [data-testid="stSidebarCollapseButton"] button {
        color: #7ab8e0 !important;
    }

    /* Header principal */
    .coia-header {
        background: linear-gradient(135deg, #0f1923 0%, #152032 60%, #0d2137 100%);
        padding: 1.8rem 2.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        border: 1px solid #1e3a52;
    }
    .coia-header h1 {
        color: #e8edf2;
        margin: 0 0 0.3rem 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .coia-header .subtitle { color: #7a9ab5; font-size: 0.87rem; margin: 0; }
    .coia-header .meta     { color: #4a6a85; font-size: 0.75rem; margin-top: 0.5rem; }

    /* Cards de modulo — alinhados com flexbox */
    .cards-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .module-card {
        flex: 1;
        background: #ffffff;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #e8ecf0;
        color: #1a2533 !important;
    }
    .module-card h4 { margin: 0 0 0.5rem 0; font-size: 0.92rem; font-weight: 600; color: inherit !important; }
    .module-card p  { margin: 0; font-size: 0.82rem; color: #444 !important; line-height: 1.5; }

    /* Titulos de secao */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #d0dae6 !important;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #2a3f55;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.1px;
    }

    /* Sidebar */
    [data-testid="stSidebarContent"] {
        background: #0f1923;
    }
    [data-testid="stSidebarContent"] * { color: #c8d8e8 !important; }
    [data-testid="stSidebarContent"] h3 {
        color: #e8edf2 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }
    [data-testid="stSidebarContent"] hr { border-color: #1e3a52 !important; }

    /* Captions */
    [data-testid="stCaptionContainer"] p { color: #8aafc8 !important; }

    /* Padding principal */
    .block-container { padding-top: 1.2rem !important; }

    /* Metricas — label visivel */
    [data-testid="stMetricLabel"] { color: #9ab5cc !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #e8edf2 !important; }

    /* Responsividade */
    @media (max-width: 768px) {
        .coia-header { padding: 1rem; }
        .coia-header h1 { font-size: 1.15rem; }
        .cards-row { flex-direction: column; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    }

    /* Botao flutuante de menu — sempre visivel */
    #coia-menu-btn {
        position: fixed;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        z-index: 99999;
        background: #1e4a6e;
        border: 1px solid #2a6a9a;
        border-left: none;
        border-radius: 0 10px 10px 0;
        padding: 14px 10px;
        cursor: pointer;
        box-shadow: 3px 0 14px rgba(0,0,0,0.55);
        display: flex;
        flex-direction: column;
        gap: 5px;
        align-items: center;
        transition: background 0.2s;
    }
    #coia-menu-btn:hover { background: #2a5f8a; }
    #coia-menu-btn span {
        display: block;
        width: 18px;
        height: 2px;
        background: #7ab8e0;
        border-radius: 2px;
    }
    #coia-menu-btn.hidden { display: none; }
</style>
""", unsafe_allow_html=True)




# =============================================================================
# Dialogs (modais)
# =============================================================================

@st.dialog("CO-IA — Como funciona", width="large")
def render_about():
    st.markdown("""
**CO-IA** detecta textos gerados por IA em um corpus de treinamento e protege modelos
de IA Generativa contra o *Model Collapse* — a degradacao progressiva causada por
treinamento recursivo sobre dados sinteticos.

---

#### O problema: Model Collapse

Quando um modelo e treinado sobre dados gerados por versoes anteriores de si mesmo,
a distribuicao estatistica do corpus se estreita a cada geracao. A "cauda longa" das
distribuicoes originais desaparece, levando a perda de diversidade e acuracia.
Shumailov et al. (2024, Nature) demonstraram que esse processo e inevitavel sem curadoria ativa.

---

#### Modulo I — Filtro de Entrada (Contamination Score)

Calcula um score de 0 a 1 por registro usando **5 estrategias estatisticas** combinadas
em ensemble ponderado:

| Estrategia | Peso | Formula / Logica |
|---|---|---|
| Marcadores LLM | 35% | `score = min(hits / 3, 1.0)` — conta expressoes tipicas de LLM (e.g. "em suma", "furthermore") |
| Uniformidade de sentencas | 20% | `CV = sigma/mu` dos comprimentos; `score = max(0, 1 - CV/0.5)` — LLMs geram sentencas mais uniformes |
| Comprimento de palavras | 18% | `score = (mu_chars - 4.5) / 3.5` — vocabulario LLM e mais formal e longo |
| Estrutura de paragrafo | 15% | Detecta inicios e conclusoes formulaicos via regex; score = proporcao de paragrafos afetados |
| Entropia de bigramas | 12% | `H = -sum(p * log2(p))`; `score = 1 - H/H_max` — LLMs repetem bigramas mais |

**Score final:** `S = 0.35*m + 0.20*u + 0.18*c + 0.15*e + 0.12*b`

Limiar padrao: **0.22** (calibrado no dataset demo — humanos max=0.18, sinteticos min=0.23).

---

#### Modulo II — Monitoramento de Degeneracao

Calcula metricas de diversidade do corpus e simula N geracoes de treinamento recursivo:

- **Entropia de Shannon (bigramas):** `H = -sum(p(x) * log2(p(x)))` — saudavel: > 5.0 bits
- **TTR (Type-Token Ratio):** `TTR = |V| / |T|` onde V = vocabulario unico, T = total de tokens — saudavel: > 0.35
- **MATTR (Moving Average TTR):** media do TTR em janelas de 100 tokens — mais estavel para textos longos
- **Risco de colapso:** media ponderada das metricas normalizadas — zonas: Seguro (<30%), Atencao (<55%), Critico (>75%)

---

#### Modulo III — Auditoria de Provenencia

- **Hash SHA-256** por registro: `h = SHA256(texto.encode('utf-8'))` — garante rastreabilidade e detecta adulteracao
- **Padrao-ouro:** subconjunto de registros humanos verificados, protegidos contra Data Poisoning
- **Reequilibrio:** recomenda proporcao humano:sintetico (padrao-ouro minimo 30%, sinteticos maximo 50%)
""")


@st.dialog("Fundamentacao Cientifica — 71 Patentes", width="large")
def render_patents():
    st.markdown("Base tecnica do CO-IA: **71 patentes** (49 Derwent Innovation + 22 INPI) mapeadas sistematicamente na disciplina MCT — UNIRIO/BSI, 2025.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Modulo I — Filtro", "Modulo II — Monitor", "Modulo III — Provenencia"])

    with tab1:
        st.markdown("""
| Codigo | Titular | Tecnica coberta |
|--------|---------|-----------------|
| WO2025037142-A1 | NEC Lab | Curadoria e qualidade de dados sinteticos |
| IN202511107978-A | Univ. Manipal | Deteccao ML de dados sinteticos (G06N) |
| US2024354648-A1 | — | Filtragem anomala de corpus (G06N 020/00) |
| CN119358696-A | — | Ensemble de features para filtragem (G06F 018/21) |
""")

    with tab2:
        st.markdown("""
| Codigo | Titular | Tecnica coberta |
|--------|---------|-----------------|
| US2025342187-A1 | Madisetti V. | Sistema multi-nivel com feedback loop |
| US2025094459-A1 | Madisetti V. | Monitoramento de geracoes recursivas |
| US2025217394-A1 | Madisetti V. | Metricas de degeneracao de modelo |
| US2024411789-A1 | Madisetti V. | Refinamento por geracao e baseline |
""")

    with tab3:
        st.markdown("""
| Codigo | Titular | Tecnica coberta |
|--------|---------|-----------------|
| US2025238634-A1 | — | Protecao de origem de dados |
| US2026080037-A1 | — | Curadoria com provenencia verificavel |
| BR 11 2026 0105 | — | Protecao de direitos autorais IA (G06F 21/16) |
| BR 11 2023 0065 | — | Rastreamento de dados compartilhados (G06N 5/02) |
""")

    st.info("As 71 patentes completas (Derwent + INPI) estao documentadas no arquivo MSL do projeto.")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### CO-IA")
        st.markdown("Curadoria de Origem para IA")
        st.markdown("---")

        st.markdown("### Dados de Entrada")
        fonte = st.radio(
            "Fonte dos dados:",
            ["Dataset de Demonstracao", "Upload CSV/JSONL"],
            index=0,
            label_visibility="collapsed",
        )

        df_input = None
        if "Demonstracao" in fonte:
            demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_dataset.csv")
            df_input = pd.read_csv(demo_path)
            st.success(f"{len(df_input)} registros carregados")
            st.caption("24 textos rotulados: 12 humanos + 12 sinteticos.")
            with open(demo_path, "rb") as _f:
                st.download_button(
                    "Baixar dataset de demonstracao",
                    _f.read(), "demo_dataset.csv", "text/csv",
                    use_container_width=True,
                )
        else:
            arquivo = st.file_uploader(
                "Arquivo:",
                type=["csv", "jsonl"],
                help="CSV com coluna obrigatoria 'texto'.",
                label_visibility="collapsed",
            )
            if arquivo:
                if arquivo.name.endswith(".jsonl"):
                    import json
                    linhas = [json.loads(l) for l in arquivo.read().decode().split("\n") if l.strip()]
                    df_input = pd.DataFrame(linhas)
                else:
                    df_input = pd.read_csv(arquivo)
                if "texto" not in df_input.columns:
                    st.error("O arquivo precisa ter a coluna 'texto'.")
                    df_input = None
                else:
                    st.success(f"{len(df_input)} registros carregados")
            else:
                st.info("Faca o upload de um arquivo CSV ou JSONL.")

        st.markdown("---")
        st.markdown("### Configuracoes")

        limiar = st.slider(
            "Limiar de classificacao:",
            min_value=0.10, max_value=0.90,
            value=DEFAULT_THRESHOLD, step=0.05,
            help="Score >= limiar = sintetico. Padrao calibrado: 0.22",
        )
        n_geracoes = st.slider(
            "Geracoes na simulacao:",
            min_value=3, max_value=15, value=8,
            help="Numero de geracoes recursivas a simular no Modulo II.",
        )

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Como funciona", use_container_width=True):
                st.session_state["show_about"] = True
        with col_b:
            if st.button("71 Patentes", use_container_width=True):
                st.session_state["show_patents"] = True

    return df_input, limiar, n_geracoes


# =============================================================================
# Aba 0 — Visao Geral
# =============================================================================

def render_home(df: pd.DataFrame):
    st.markdown("""
    <div class="coia-header">
        <h1>CO-IA &mdash; Curadoria de Origem para IA</h1>
        <p class="subtitle">Pipeline de deteccao, monitoramento e provenencia para mitigacao do Model Collapse</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="cards-row">
        <div class="module-card" style="border-top:3px solid #EF5350;">
            <h4 style="color:#c62828 !important;">Modulo I &mdash; Filtro de Entrada</h4>
            <p>Calcula o <em>contamination score</em> (0&ndash;1) de cada texto usando 5 estrategias
            estatisticas e classifica como humano ou sintetico.</p>
        </div>
        <div class="module-card" style="border-top:3px solid #FB8C00;">
            <h4 style="color:#e65100 !important;">Modulo II &mdash; Monitoramento</h4>
            <p>Mede a diversidade do corpus (entropia, TTR, MATTR) e simula a degradacao
            ao longo de N geracoes de treinamento recursivo.</p>
        </div>
        <div class="module-card" style="border-top:3px solid #43A047;">
            <h4 style="color:#2e7d32 !important;">Modulo III &mdash; Provenencia</h4>
            <p>Gera hash SHA-256 por registro, define um padrao-ouro protegido e
            recomenda o reequilibrio humano:sintetico do corpus.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Corpus carregado</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de registros", len(df))
    if "tipo_real" in df.columns:
        c2.metric("Humanos", (df["tipo_real"] == "humano").sum())
        c3.metric("Sinteticos", (df["tipo_real"] == "sintetico").sum())
    c4.metric("Colunas", len(df.columns))

    st.markdown("**Previa dos dados:**")
    st.dataframe(df.head(6), use_container_width=True)


# =============================================================================
# Aba 1 — Modulo I
# =============================================================================

def render_modulo1(df: pd.DataFrame, limiar: float):
    st.markdown('<div class="section-title">Modulo I — Filtro de Entrada</div>', unsafe_allow_html=True)
    st.caption("Ensemble de 5 estrategias estatisticas ponderadas. Score >= limiar = sintetico.")

    engine = build_default_engine(limiar=limiar)

    with st.spinner("Calculando contamination scores..."):
        df_resultado = engine.analisar_dataset(df)

    total      = len(df_resultado)
    n_sint     = int((df_resultado["classificacao_coia"] == "sintetico").sum())
    n_hum      = total - n_sint
    score_med  = round(df_resultado["score_contaminacao"].mean(), 3)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analisado", total)
    c2.metric("Humanos", n_hum)
    c3.metric("Sinteticos", n_sint)
    c4.metric("Score medio", score_med, delta=f"limiar {limiar}", delta_color="off")

    if "tipo_real" in df_resultado.columns:
        df_resultado["correto"] = df_resultado.apply(
            lambda r: r["classificacao_coia"] == r["tipo_real"] or
                      (r["classificacao_coia"] == "sintetico" and r["tipo_real"] == "sintetico"), axis=1)
        acc = df_resultado["correto"].mean() * 100
        st.metric("Acuracia vs rotulo real", f"{acc:.1f}%")

    st.markdown("#### Distribuicao do Contamination Score")
    fig_hist = px.histogram(
        df_resultado, x="score_contaminacao", color="classificacao_coia", nbins=20,
        color_discrete_map={"humano": COLOR_HUMANO, "sintetico": COLOR_SINTETICO},
        labels={"score_contaminacao": "Contamination Score", "classificacao_coia": "Classificacao CO-IA"},
    )
    fig_hist.add_vline(x=limiar, line_dash="dash", line_color="#1E88E5",
                       annotation_text=f"Limiar ({limiar})", annotation_position="top right")
    fig_hist.update_layout(height=320, margin=dict(t=30, b=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, use_container_width=True)

    score_cols = [c for c in df_resultado.columns if c.startswith("score_") and c != "score_contaminacao"]
    if score_cols:
        st.markdown("#### Perfil por estrategia")
        medias = df_resultado.groupby("classificacao_coia")[score_cols].mean().reset_index()
        estrategias = [c.replace("score_", "").replace("_", " ").title() for c in score_cols]
        fig_radar = go.Figure()
        for _, row in medias.iterrows():
            valores = [row[c] for c in score_cols] + [row[score_cols[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=estrategias + [estrategias[0]],
                fill="toself",
                name=row["classificacao_coia"],
                line_color=COLOR_HUMANO if row["classificacao_coia"] == "humano" else COLOR_SINTETICO,
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=340, title="Score medio por classe",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### Tabela de resultados")
    colunas = ["id", "texto", "score_contaminacao", "classificacao_coia"] + score_cols[:3]
    colunas = [c for c in colunas if c in df_resultado.columns]
    st.dataframe(df_resultado[colunas], use_container_width=True, height=380)

    st.session_state["df_m1"] = df_resultado
    return df_resultado


# =============================================================================
# Aba 2 — Modulo II
# =============================================================================

def render_modulo2(df: pd.DataFrame, n_geracoes: int):
    st.markdown('<div class="section-title">Modulo II — Monitoramento de Degeneracao</div>', unsafe_allow_html=True)
    st.caption("Metricas de diversidade e simulacao de treinamento recursivo por N geracoes.")

    textos = df["texto"].tolist() if "texto" in df.columns else []
    df_m1  = st.session_state.get("df_m1", df)

    with st.spinner("Calculando metricas e simulando geracoes..."):
        resultado = analisar_corpus(textos, df=df_m1, n_geracoes=n_geracoes)

    metricas = resultado["metricas"]
    risco    = resultado["risco"]
    sim      = resultado["simulacao"]

    risco_val = risco["risco_total"]
    cor_gauge = (COLOR_HUMANO if risco_val < COLLAPSE_RISK_THRESHOLDS["seguro"]
                 else COLOR_ALERTA if risco_val < COLLAPSE_RISK_THRESHOLDS["atencao"]
                 else COLOR_SINTETICO)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risco_val * 100,
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": f"Risco de Colapso de Modelo — {risco['zona']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": cor_gauge},
            "steps": [
                {"range": [0,  30], "color": "#0a2a0a"},
                {"range": [30, 55], "color": "#2a1e00"},
                {"range": [55, 75], "color": "#2a1500"},
                {"range": [75, 100],"color": "#2a0a0a"},
            ],
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=60, b=10),
                             paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Entropia (bigramas)", f"{metricas['entropia_bigramas']:.3f}",
               delta="saudavel > 5.0", delta_color="normal" if metricas['entropia_bigramas'] > 5.0 else "inverse")
    mc2.metric("TTR do corpus", f"{metricas['ttr']:.3f}",
               delta="saudavel > 0.35", delta_color="normal" if metricas['ttr'] > 0.35 else "inverse")
    mc3.metric("MATTR", f"{metricas['mattr']:.3f}")
    mc4.metric("Prop. sinteticos", f"{metricas['proporcao_sinteticos']*100:.1f}%")

    st.markdown("#### Simulacao de geracoes de treinamento recursivo")
    tab_e, tab_t, tab_r = st.tabs(["Entropia", "TTR", "Risco de Colapso"])

    kwargs = dict(x="geracao", color="cenario",
                  color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
                  markers=True)
    layout_kw = dict(height=300, margin=dict(t=30, b=20),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    with tab_e:
        fig = px.line(sim, y="entropia_bigramas", **kwargs)
        fig.add_hline(y=5.0, line_dash="dot", line_color=COLOR_NEUTRO,
                      annotation_text="Minimo saudavel")
        fig.update_layout(**layout_kw)
        st.plotly_chart(fig, use_container_width=True)

    with tab_t:
        fig = px.line(sim, y="ttr", **kwargs)
        fig.add_hline(y=0.35, line_dash="dot", line_color=COLOR_NEUTRO,
                      annotation_text="Minimo saudavel")
        fig.update_layout(**layout_kw)
        st.plotly_chart(fig, use_container_width=True)

    with tab_r:
        fig = px.line(sim, y="risco_colapso", **kwargs)
        for nome, val in COLLAPSE_RISK_THRESHOLDS.items():
            fig.add_hline(y=val, line_dash="dash", line_color=COLOR_ALERTA,
                          annotation_text=nome)
        fig.update_layout(**layout_kw)
        st.plotly_chart(fig, use_container_width=True)

    st.session_state["metricas_m2"] = metricas
    st.session_state["risco_m2"]    = risco
    return metricas, risco


# =============================================================================
# Aba 3 — Modulo III
# =============================================================================

def render_modulo3(df: pd.DataFrame):
    st.markdown('<div class="section-title">Modulo III — Auditoria de Provenencia</div>', unsafe_allow_html=True)
    st.caption("Hash SHA-256 por registro, padrao-ouro e recomendacao de reequilibrio do corpus.")

    df_m1 = st.session_state.get("df_m1", df)
    if "classificacao_coia" not in df_m1.columns:
        engine = build_default_engine()
        df_m1  = engine.analisar_dataset(df_m1)

    st.markdown("#### Padrao-ouro (Gold Standard)")
    st.caption("Marque os registros humanos verificados. Eles serao protegidos contra Data Poisoning.")

    candidatos = df_m1[df_m1["classificacao_coia"] == "humano"]["id"].tolist() \
        if "id" in df_m1.columns else []
    ids_gold = st.multiselect(
        "Selecione os IDs para o padrao-ouro:",
        options=candidatos,
        default=candidatos[:min(4, len(candidatos))],
        label_visibility="collapsed",
    )

    with st.spinner("Gerando registros de provenencia (SHA-256)..."):
        resultado_prov = gerar_proveniencia(df_m1, ids_gold=ids_gold)

    req    = resultado_prov["reequilibrio"]
    df_prov = resultado_prov["df_provenance"]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Humanos",    f"{req['proporcao_humanos']}%",    f"{req['n_humanos']} reg.")
    e2.metric("Sinteticos", f"{req['proporcao_sinteticos']}%", f"{req['n_sinteticos']} reg.")
    e3.metric("Padrao-ouro",f"{req['proporcao_gold']}%",       f"{req['n_gold_standard']} reg.")
    e4.metric("Limite sint.",f"<= {req['limite_sinteticos']:.0f}%")

    col_pie, col_status = st.columns([1, 1])
    with col_pie:
        fig = px.pie(
            values=[req["n_humanos"], req["n_sinteticos"]],
            names=["Humanos", "Sinteticos"],
            color_discrete_sequence=[COLOR_HUMANO, COLOR_SINTETICO],
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=250, margin=dict(t=20, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_status:
        st.markdown("**Recomendacao de reequilibrio:**")
        status = req.get("status", "")
        acao   = req.get("acao_recomendada", "")
        if "OK" in status or "Equilibrado" in status:
            st.success(f"{status}\n\n{acao}")
        else:
            st.warning(f"{status}\n\n{acao}")

    st.markdown("#### Registros de Provenencia (SHA-256)")
    if not df_prov.empty:
        df_exib = df_prov.copy()
        if "hash_sha256" in df_exib.columns:
            df_exib["hash_sha256"] = df_exib["hash_sha256"].str[:22] + "..."
        cols = ["record_id", "hash_sha256", "fonte", "tipo_declarado",
                "score_contaminacao", "is_gold_standard", "comprimento_chars"]
        st.dataframe(df_exib[[c for c in cols if c in df_exib.columns]],
                     use_container_width=True, height=340)

    st.session_state["reequilibrio"] = req
    st.session_state["df_prov"]      = df_prov
    return req, df_prov


# =============================================================================
# Aba 4 — Exportar Relatorio
# =============================================================================

def render_exportacao():
    st.markdown('<div class="section-title">Exportar Relatorio de Auditoria</div>', unsafe_allow_html=True)
    st.caption("Execute os tres modulos antes de exportar para obter o relatorio completo.")

    m1  = st.session_state.get("metricas_m1",  {"total_registros": 0, "n_humanos": 0, "n_sinteticos": 0, "taxa_contaminacao": 0.0, "score_medio": 0.0})
    m2  = st.session_state.get("risco_m2",     {"zona": "N/A", "risco_total": 0.0})
    req = st.session_state.get("reequilibrio", {"status": "—"})
    dfp = st.session_state.get("df_prov",      pd.DataFrame())

    relatorio = (
        ReportBuilder()
        .adicionar_cabecalho(APP_TITLE, APP_SUBTITLE, APP_VERSION)
        .adicionar_resumo_execucao(m1, m2, req)
        .adicionar_registros_proveniencia(dfp)
        .adicionar_reequilibrio(req)
        .adicionar_fundamentacao()
        .adicionar_rodape()
        .build()
    )

    with st.expander("Previa do relatorio", expanded=True):
        st.markdown(relatorio)

    st.download_button(
        "Baixar relatorio (.md)",
        relatorio.encode("utf-8"),
        "relatorio_co_ia.md", "text/markdown",
        use_container_width=True,
    )

    st.markdown("#### Resumo para registro INPI/DIT")
    resumo = f"""Nome: CO-IA — Curadoria de Origem para IA
Versao: {APP_VERSION}
Linguagem: Python 3.11 | Plataforma: Streamlit

Finalidade: Mitigacao do Model Collapse em sistemas de IA Generativa por curadoria
integrada de origem dos dados de treinamento. Tres modulos: deteccao estatistica
(contamination score), monitoramento de diversidade e simulacao de geracoes, e
auditoria de provenencia com hashing criptografico SHA-256.

Embasamento: 71 patentes (Derwent Innovation + INPI) e 244 estudos cientificos.

Autor: Gabriel de Melo Guedes Souza — UNIRIO/BSI
Disciplina: Metodologia Cientifica e Tecnologica (MCT)
"""
    with st.expander("Ver resumo INPI/DIT"):
        st.code(resumo, language=None)
    st.download_button(
        "Baixar resumo INPI (.txt)",
        resumo.encode("utf-8"),
        "resumo_inpi_co_ia.txt", "text/plain",
        use_container_width=True,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    # Modais
    if st.session_state.get("show_about"):
        st.session_state["show_about"] = False
        render_about()
    if st.session_state.get("show_patents"):
        st.session_state["show_patents"] = False
        render_patents()

    df_input, limiar, n_geracoes = render_sidebar()

    # Botao flutuante para reabrir a sidebar
    st.markdown("""
<div id="coia-menu-btn" title="Abrir menu lateral">
  <span></span><span></span><span></span>
</div>
<script>
(function() {
  function clickSidebarToggle() {
    var collapsed = document.querySelector('[data-testid="collapsedControl"]');
    if (collapsed) { collapsed.click(); return; }
    var closeBtn = document.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (closeBtn) { closeBtn.click(); return; }
  }
  function syncVisibility() {
    var btn = document.getElementById('coia-menu-btn');
    if (!btn) return;
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
      btn.style.display = sidebar.getBoundingClientRect().width > 50 ? 'none' : 'flex';
    }
  }
  setTimeout(function() {
    var btn = document.getElementById('coia-menu-btn');
    if (btn) btn.addEventListener('click', clickSidebarToggle);
    setInterval(syncVisibility, 400);
  }, 200);
})();
</script>
""", unsafe_allow_html=True)

    if df_input is None:
        st.markdown("""
        <div class="coia-header">
            <h1>CO-IA — Curadoria de Origem para IA</h1>
            <p class="subtitle">Selecione o dataset de demonstracao na barra lateral para comecar.</p>
            <p class="meta">Use o botao "Como funciona" na barra lateral para entender o sistema.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Selecione 'Dataset de Demonstracao' na barra lateral (esquerda) para ver o sistema em acao com 24 textos rotulados.")
        return

    if "id" not in df_input.columns:
        df_input["id"] = [f"REC_{i:04d}" for i in range(len(df_input))]

    abas = st.tabs([
        "Visao Geral",
        "Modulo I — Filtro",
        "Modulo II — Monitor",
        "Modulo III — Provenencia",
        "Exportar Relatorio",
    ])

    with abas[0]:
        render_home(df_input)
    with abas[1]:
        render_modulo1(df_input, limiar)
    with abas[2]:
        render_modulo2(st.session_state.get("df_m1", df_input), n_geracoes)
    with abas[3]:
        render_modulo3(df_input)
    with abas[4]:
        render_exportacao()


if __name__ == "__main__":
    main()
