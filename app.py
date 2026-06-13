# =============================================================================
# CO-IA — Curadoria de Origem para IA
# app.py — Dashboard principal (Streamlit)
# =============================================================================

import io
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    APP_AUTHOR,
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    COLOR_ALERTA,
    COLOR_HUMANO,
    COLOR_INFO,
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
# Configuração de página
# =============================================================================

st.set_page_config(
    page_title="CO-IA",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tipografia base */
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

    /* Header principal */
    .coia-header {
        background: linear-gradient(135deg, #0f1923 0%, #152032 60%, #0d2137 100%);
        padding: 2rem 2.5rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #1e3a52;
    }
    .coia-header h1 {
        color: #e8edf2;
        margin: 0 0 0.4rem 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    .coia-header .subtitle {
        color: #7a9ab5;
        font-size: 0.9rem;
        margin: 0;
    }
    .coia-header .meta {
        color: #4a6a85;
        font-size: 0.78rem;
        margin-top: 0.6rem;
    }

    /* Cards de módulo */
    .module-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e8ecf0;
        height: 100%;
    }
    .module-card h4 { margin: 0 0 0.5rem 0; font-size: 0.95rem; font-weight: 600; }
    .module-card p  { margin: 0; font-size: 0.85rem; color: #555; line-height: 1.5; }

    /* Títulos de seção */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a2533;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e8ecf0;
        margin: 1.8rem 0 1.1rem 0;
        letter-spacing: -0.2px;
    }

    /* Badge de patente */
    .patent-badge {
        display: inline-block;
        background: #f0f5fa;
        border: 1px solid #c5d8e8;
        color: #2c5282;
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 4px;
        letter-spacing: 0.2px;
    }

    /* Rótulos de risco */
    .risk-safe     { background:#f0faf2; border-left:3px solid #43A047; padding:0.7rem 1rem; border-radius:6px; }
    .risk-warning  { background:#fffbf0; border-left:3px solid #FB8C00; padding:0.7rem 1rem; border-radius:6px; }
    .risk-critical { background:#fff5f5; border-left:3px solid #EF5350; padding:0.7rem 1rem; border-radius:6px; }

    /* Sidebar limpa */
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebarContent"] { background: #0f1923; }
    [data-testid="stSidebarContent"] * { color: #c8d8e8 !important; }
    [data-testid="stSidebarContent"] h3 { color: #e8edf2 !important; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stSidebarContent"] .stSlider label { font-size: 0.8rem; }

    /* Remover padding excessivo */
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### CO-IA — Curadoria de Origem")
        st.markdown("---")

        st.markdown("### Dados de Entrada")
        fonte = st.radio(
            "Fonte:",
            ["Dataset de Demonstração", "Upload CSV/JSONL"],
            index=0,
            label_visibility="collapsed",
        )

        df_input = None
        if "Dataset de Demonstração" in fonte:
            demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_dataset.csv")
            df_input = pd.read_csv(demo_path)
            st.success(f"Demo carregada — {len(df_input)} registros")
            st.caption("12 textos humanos + 12 sintéticos rotulados.")
        else:
            arquivo = st.file_uploader(
                "Arquivo:",
                type=["csv", "jsonl"],
                help="CSV com coluna obrigatória 'texto'.",
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
                    st.error("O arquivo deve conter a coluna 'texto'.")
                    df_input = None
                else:
                    st.success(f"Carregado — {len(df_input)} registros")
            else:
                st.info("Aguardando arquivo...")

        st.markdown("---")
        st.markdown("### Configurações")

        limiar = st.slider(
            "Limiar de classificação:",
            min_value=0.10, max_value=0.90,
            value=DEFAULT_THRESHOLD, step=0.05,
            help="Score >= limiar → sintético. Ref: IN202511107978-A",
        )
        n_geracoes = st.slider(
            "Geracoes na simulação:",
            min_value=3, max_value=15, value=8,
            help="US2025094459-A1 (Madisetti)",
        )

        st.markdown("---")

        # Botão "Sobre"
        if st.button("Sobre o CO-IA", use_container_width=True):
            st.session_state["show_about"] = True

    return df_input, limiar, n_geracoes


# =============================================================================
# Modal "Sobre o CO-IA"
# =============================================================================

@st.dialog("Sobre o CO-IA", width="large")
def render_about():
    st.markdown("""
**CO-IA — Curadoria de Origem para IA** é um framework integrado para mitigação do
*Model Collapse* em sistemas de Inteligência Artificial Generativa.

---

**O problema**

Quando modelos de IA Generativa são treinados recursivamente sobre dados sintéticos
produzidos por versões anteriores do próprio modelo, ocorre uma degradação progressiva
em acurácia e diversidade — o chamado *Model Collapse* (efeito *"Habsburg AI"*).
A perda da "cauda longa" da distribuição dos dados originais compromete a utilidade
dos modelos ao longo das gerações.

**A solução — os três módulos**

| Módulo | Função |
|--------|--------|
| **I — Filtro de Entrada** | Calcula um *contamination score* (0–1) por registro e classifica textos como humanos ou sintéticos usando 5 estratégias estatísticas explicáveis |
| **II — Monitoramento** | Calcula métricas de diversidade do corpus (entropia, TTR, MATTR) e simula a degradação ao longo de N gerações com e sem o CO-IA ativo |
| **III — Proveniência** | Gera hash SHA-256 por registro, protege um conjunto padrão-ouro contra *Data Poisoning* e recomenda o reequilíbrio humano:sintético |

**Base científica**

Fundamentado em **71 patentes** (Derwent Innovation + INPI) e **244 estudos científicos**
mapeados sistematicamente no âmbito da disciplina Metodologia Científica e Tecnológica
(MCT) — UNIRIO / Bacharelado em Sistemas de Informação, 2025.

**Principais patentes de referência:** WO2025037142-A1 (NEC Lab) · IN202511107978-A (Univ. Manipal) ·
US2025342187-A1 (Madisetti) · US2025238634-A1 · BR 11 2026 0105 · BR 11 2023 0065

---

**Autor:** Gabriel de Melo Guedes Souza — UNIRIO / BSI
**Orientadora:** Prof.ª Maria Augusta Silveira Netto Nunes
**Versão:** {ver}
**Repositório:** [github.com/gbmelo30-boop/co-ia](https://github.com/gbmelo30-boop/co-ia)
""".format(ver=APP_VERSION))


# =============================================================================
# Aba 0 — Visão Geral
# =============================================================================

def render_home(df: pd.DataFrame):
    st.markdown("""
    <div class="coia-header">
        <h1>CO-IA — Curadoria de Origem para IA</h1>
        <p class="subtitle">Pipeline integrado de detecção, monitoramento e proveniência para mitigação do Model Collapse</p>
        <p class="meta">UNIRIO / BSI &nbsp;·&nbsp; Metodologia Científica e Tecnológica &nbsp;·&nbsp; 2025 &nbsp;|&nbsp;
        Fundamentado em 71 patentes (Derwent + INPI) e 244 estudos científicos</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="module-card" style="border-top:3px solid #EF5350;">
            <h4 style="color:#c62828;">Módulo I — Filtro de Entrada</h4>
            <p>Calcula <em>contamination score</em> (0–1) e classifica cada texto como humano
            ou sintético usando 5 estratégias estatísticas explicáveis.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="module-card" style="border-top:3px solid #FB8C00;">
            <h4 style="color:#e65100;">Módulo II — Monitoramento</h4>
            <p>Calcula métricas de diversidade do corpus, exibe o risco de colapso e
            simula a degradação ao longo de N gerações de treinamento recursivo.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="module-card" style="border-top:3px solid #43A047;">
            <h4 style="color:#2e7d32;">Módulo III — Proveniência</h4>
            <p>Gera hash SHA-256 por registro, protege o padrão-ouro e recomenda
            o reequilíbrio humano:sintético do corpus.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Visão Rápida do Corpus Carregado</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Registros", len(df))
    if "tipo_real" in df.columns:
        c2.metric("Humanos (rótulo real)", (df["tipo_real"] == "humano").sum())
        c3.metric("Sintéticos (rótulo real)", (df["tipo_real"] == "sintetico").sum())
    c4.metric("Colunas disponíveis", len(df.columns))

    st.markdown("**Prévia dos dados:**")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown('<div class="section-title">Fundamentação — Patentes por Módulo</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    for col, modulo, label in [(pc1, "modulo1", "Módulo I — Filtro"),
                                (pc2, "modulo2", "Módulo II — Monitor"),
                                (pc3, "modulo3", "Módulo III — Proveniência")]:
        with col:
            st.markdown(f"**{label}**")
            for p in PATENT_REFERENCES[modulo]:
                st.markdown(
                    f'<span class="patent-badge">{p["id"]}</span> {p["tecnica"]}',
                    unsafe_allow_html=True,
                )
                st.markdown("")


# =============================================================================
# Aba 1 — Módulo I
# =============================================================================

def render_modulo1(df: pd.DataFrame, limiar: float):
    st.markdown('<div class="section-title">Módulo I — Filtro de Entrada (Contamination Score)</div>', unsafe_allow_html=True)
    st.caption("Ref: WO2025037142-A1 · IN202511107978-A · CN119358696-A · US2024354648-A1")

    engine = build_default_engine(limiar=limiar)

    with st.spinner("Calculando contamination scores..."):
        df_resultado = engine.analisar_dataset(df)

    # metricas_resumo inline para evitar cache
    total        = len(df_resultado)
    n_sint       = int((df_resultado["classificacao_coia"] == "sintético").sum())
    n_hum        = total - n_sint
    metricas = {
        "total_registros":   total,
        "n_humanos":         n_hum,
        "n_sinteticos":      n_sint,
        "taxa_contaminacao": round(n_sint / total * 100, 1) if total > 0 else 0.0,
        "score_medio":       round(df_resultado["score_contaminacao"].mean(), 3),
        "score_max":         round(df_resultado["score_contaminacao"].max(), 3),
        "score_min":         round(df_resultado["score_contaminacao"].min(), 3),
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analisado", metricas["total_registros"])
    c2.metric("Humanos", metricas["n_humanos"])
    c3.metric("Sintéticos", metricas["n_sinteticos"])
    c4.metric("Score Médio", metricas["score_medio"], delta=f"limiar: {limiar}", delta_color="off")

    st.markdown("#### Distribuição do Contamination Score")
    fig_hist = px.histogram(
        df_resultado, x="score_contaminacao", color="classificacao_coia", nbins=20,
        color_discrete_map={"humano": COLOR_HUMANO, "sintético": COLOR_SINTETICO},
        labels={"score_contaminacao": "Contamination Score", "classificacao_coia": "Classificação CO-IA"},
        title="Histograma de Scores — Módulo I",
    )
    fig_hist.add_vline(x=limiar, line_dash="dash", line_color="#1E88E5",
                       annotation_text=f"Limiar ({limiar})", annotation_position="top right")
    fig_hist.update_layout(height=340, margin=dict(t=50, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    score_cols = [c for c in df_resultado.columns if c.startswith("score_") and c != "score_contaminacao"]
    if score_cols:
        st.markdown("#### Perfil de Scores por Estratégia")
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
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                 height=360, title="Perfil médio por classe")
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### Tabela de Resultados")
    colunas = ["id", "texto", "score_contaminacao", "classificacao_coia"] + score_cols[:3]
    colunas = [c for c in colunas if c in df_resultado.columns]

    def colorir(row):
        cor = "#fff0f0" if row["classificacao_coia"] == "sintético" else "#f0fff4"
        return [f"background-color: {cor}"] * len(row)

    st.dataframe(df_resultado[colunas].style.apply(colorir, axis=1),
                 use_container_width=True, height=400)

    if "tipo_real" in df_resultado.columns:
        df_resultado["correto"] = df_resultado.apply(
            lambda r: r["classificacao_coia"] == r["tipo_real"] or
                      (r["classificacao_coia"] == "sintético" and r["tipo_real"] == "sintetico"), axis=1)
        acc = df_resultado["correto"].mean() * 100
        st.metric("Acurácia vs rótulo real", f"{acc:.1f}%")

    st.session_state["df_m1"]      = df_resultado
    st.session_state["metricas_m1"] = metricas
    return df_resultado


# =============================================================================
# Aba 2 — Módulo II
# =============================================================================

def render_modulo2(df: pd.DataFrame, n_geracoes: int):
    st.markdown('<div class="section-title">Módulo II — Monitoramento de Degeneração</div>', unsafe_allow_html=True)
    st.caption("Ref: US2025342187-A1 · US2025094459-A1 · US2025217394-A1 · US2024411789-A1 (Madisetti V.)")

    textos = df["texto"].tolist() if "texto" in df.columns else []
    df_m1  = st.session_state.get("df_m1", df)

    with st.spinner("Calculando métricas de diversidade e risco..."):
        resultado = analisar_corpus(textos, df=df_m1, n_geracoes=n_geracoes)

    metricas = resultado["metricas"]
    risco    = resultado["risco"]
    sim      = resultado["simulacao"]

    risco_val = risco["risco_total"]
    cor_gauge = (COLOR_HUMANO if risco_val < COLLAPSE_RISK_THRESHOLDS["seguro"]
                 else COLOR_ALERTA if risco_val < COLLAPSE_RISK_THRESHOLDS["atencao"]
                 else COLOR_SINTETICO)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risco_val * 100,
        number={"suffix": "%", "font": {"size": 30}},
        title={"text": f"Risco de Colapso de Modelo<br><sub>{risco['zona']}</sub>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": cor_gauge},
            "steps": [
                {"range": [0,  30], "color": "#f0faf2"},
                {"range": [30, 55], "color": "#fffbf0"},
                {"range": [55, 75], "color": "#fff3e0"},
                {"range": [75, 100],"color": "#fff5f5"},
            ],
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Entropia (bigramas)", f"{metricas['entropia_bigramas']:.3f}", help="Saudável: > 5.0")
    mc2.metric("TTR do Corpus",       f"{metricas['ttr']:.3f}",              help="Saudável: > 0.35")
    mc3.metric("MATTR",               f"{metricas['mattr']:.3f}")
    mc4.metric("Prop. Sintéticos",    f"{metricas['proporcao_sinteticos']*100:.1f}%")

    st.markdown("#### Simulação de Gerações de Treinamento Recursivo")
    st.caption("Baseado em US2025342187-A1 e US2025094459-A1 (Madisetti) · Shumailov et al., 2024")

    tab_e, tab_t, tab_r = st.tabs(["Entropia", "TTR", "Risco de Colapso"])

    with tab_e:
        fig = px.line(sim, x="geracao", y="entropia_bigramas", color="cenario",
                      color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
                      markers=True, title="Entropia ao longo das gerações")
        fig.add_hline(y=5.0, line_dash="dot", line_color=COLOR_NEUTRO, annotation_text="Mínimo saudável")
        st.plotly_chart(fig, use_container_width=True)

    with tab_t:
        fig = px.line(sim, x="geracao", y="ttr", color="cenario",
                      color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
                      markers=True, title="TTR ao longo das gerações")
        fig.add_hline(y=0.35, line_dash="dot", line_color=COLOR_NEUTRO, annotation_text="Mínimo saudável")
        st.plotly_chart(fig, use_container_width=True)

    with tab_r:
        fig = px.line(sim, x="geracao", y="risco_colapso", color="cenario",
                      color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
                      markers=True, title="Evolução do risco de colapso")
        for nome, val in COLLAPSE_RISK_THRESHOLDS.items():
            fig.add_hline(y=val, line_dash="dash", line_color=COLOR_ALERTA, annotation_text=nome)
        st.plotly_chart(fig, use_container_width=True)

    st.session_state["metricas_m2"] = metricas
    st.session_state["risco_m2"]    = risco
    return metricas, risco


# =============================================================================
# Aba 3 — Módulo III
# =============================================================================

def render_modulo3(df: pd.DataFrame):
    st.markdown('<div class="section-title">Módulo III — Auditoria de Proveniência</div>', unsafe_allow_html=True)
    st.caption("Ref: US2025238634-A1 · US2026080037-A1 · BR 11 2026 0105 (G06F 21/16) · BR 11 2023 0065 (G06N 5/02)")

    df_m1 = st.session_state.get("df_m1", df)
    if "classificacao_coia" not in df_m1.columns:
        engine = build_default_engine()
        df_m1  = engine.analisar_dataset(df_m1)

    st.markdown("#### Definir Padrão-Ouro (Gold Standard)")
    st.caption("Registros humanos verificados protegidos contra Data Poisoning. Ref: US2025238634-A1")

    candidatos = df_m1[df_m1["classificacao_coia"] == "humano"]["id"].tolist() if "id" in df_m1.columns else []
    ids_gold = st.multiselect(
        "IDs para padrão-ouro:",
        options=candidatos,
        default=candidatos[:min(4, len(candidatos))],
    )

    with st.spinner("Gerando registros de proveniência (SHA-256)..."):
        resultado_prov = gerar_proveniencia(df_m1, ids_gold=ids_gold)

    req    = resultado_prov["reequilibrio"]
    df_prov = resultado_prov["df_provenance"]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Humanos",       f"{req['proporcao_humanos']}%",   f"{req['n_humanos']} reg.")
    e2.metric("Sintéticos",    f"{req['proporcao_sinteticos']}%", f"{req['n_sinteticos']} reg.")
    e3.metric("Padrão-Ouro",   f"{req['proporcao_gold']}%",       f"{req['n_gold_standard']} reg.")
    e4.metric("Limite sintét.", f"<= {req['limite_sinteticos']:.0f}%")

    fig = px.pie(
        values=[req["n_humanos"], req["n_sinteticos"]],
        names=["Humanos", "Sintéticos"],
        color_discrete_sequence=[COLOR_HUMANO, COLOR_SINTETICO],
        title="Composição do Corpus", hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=280, margin=dict(t=50, b=5))
    st.plotly_chart(fig, use_container_width=True)

    status = req["status"]
    if "✅" in status:
        st.success(f"**{status}**\n\n{req['acao_recomendada']}")
    elif "⚠️" in status:
        st.error(f"**{status}**\n\n{req['acao_recomendada']}")
    else:
        st.warning(f"**{status}**\n\n{req['acao_recomendada']}")

    st.markdown("#### Registros de Proveniência (SHA-256)")
    if not df_prov.empty:
        df_exib = df_prov.copy()
        if "hash_sha256" in df_exib.columns:
            df_exib["hash_sha256"] = df_exib["hash_sha256"].str[:20] + "..."
        cols = ["record_id", "hash_sha256", "fonte", "tipo_declarado",
                "score_contaminacao", "is_gold_standard", "comprimento_chars"]
        st.dataframe(df_exib[[c for c in cols if c in df_exib.columns]],
                     use_container_width=True, height=340)

    st.session_state["reequilibrio"] = req
    st.session_state["df_prov"]      = df_prov
    return req, df_prov


# =============================================================================
# Aba 4 — Exportação
# =============================================================================

def render_exportacao():
    st.markdown('<div class="section-title">Exportar Relatório de Auditoria</div>', unsafe_allow_html=True)
    st.caption("Relatório Markdown exportável — evidência para pitch acadêmico e registro INPI/DIT")

    m1  = st.session_state.get("metricas_m1",  {"total_registros": 0, "n_humanos": 0, "n_sinteticos": 0, "taxa_contaminacao": 0.0, "score_medio": 0.0})
    m2  = st.session_state.get("risco_m2",     {"zona": "N/A", "risco_total": 0.0})
    req = st.session_state.get("reequilibrio", {"status": "Execute os módulos primeiro."})
    dfp = st.session_state.get("df_prov",      pd.DataFrame())

    builder = (
        ReportBuilder()
        .adicionar_cabecalho(APP_TITLE, APP_SUBTITLE, APP_VERSION)
        .adicionar_resumo_execucao(m1, m2, req)
        .adicionar_registros_proveniencia(dfp)
        .adicionar_reequilibrio(req)
        .adicionar_fundamentacao()
        .adicionar_rodape()
    )
    relatorio = builder.build()

    with st.expander("Prévia do relatório", expanded=True):
        st.markdown(relatorio)

    st.download_button("Baixar relatório (.md)", relatorio.encode("utf-8"),
                       "relatorio_co_ia.md", "text/markdown")

    st.markdown("#### Resumo Técnico — Registro de Software (INPI/DIT)")
    resumo = f"""Nome do Programa: CO-IA — Curadoria de Origem para IA
Versão: {APP_VERSION}
Linguagem: Python 3.11
Plataforma: Streamlit (web) — CPU only

Finalidade: Software para mitigação do Colapso de Modelos (Model Collapse) em sistemas
de IA Generativa, por meio da curadoria integrada de origem dos dados de treinamento.
Combina detecção estatística de dados sintéticos (Módulo I), monitoramento de diversidade
e risco do corpus (Módulo II) e auditoria de proveniência com hashing criptográfico (Módulo III).

Módulos:
  1. Filtro de Entrada — ensemble de 5 estratégias de pontuação (contamination score 0-1)
  2. Monitoramento de Degeneração — métricas Shannon/TTR/MATTR + simulação de gerações
  3. Auditoria de Proveniência — SHA-256, padrão-ouro, reequilíbrio de corpus

Embasamento técnico: 71 patentes (Derwent Innovation + INPI) e 244 estudos científicos
mapeados sistematicamente — MCT / UNIRIO / BSI.

Autor: Gabriel de Melo Guedes Souza
Instituição: UNIRIO — Bacharelado em Sistemas de Informação (BSI)
"""
    with st.expander("Ver resumo INPI/DIT"):
        st.code(resumo, language=None)

    st.download_button("Baixar resumo INPI (.txt)", resumo.encode("utf-8"),
                       "resumo_inpi_co_ia.txt", "text/plain")


# =============================================================================
# Main
# =============================================================================

def main():
    # Modal "Sobre"
    if st.session_state.get("show_about"):
        st.session_state["show_about"] = False
        render_about()

    df_input, limiar, n_geracoes = render_sidebar()

    if df_input is None:
        st.markdown("""
        <div class="coia-header">
            <h1>CO-IA — Curadoria de Origem para IA</h1>
            <p class="subtitle">Carregue um dataset na barra lateral ou use o dataset de demonstração para começar.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Selecione 'Dataset de Demonstração' na barra lateral para ver o sistema em ação.")
        return

    if "id" not in df_input.columns:
        df_input["id"] = [f"REC_{i:04d}" for i in range(len(df_input))]

    abas = st.tabs([
        "Visão Geral",
        "Módulo I — Filtro",
        "Módulo II — Monitor",
        "Módulo III — Proveniência",
        "Exportar Relatório",
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
