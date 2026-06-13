# =============================================================================
# CO-IA — Curadoria de Origem para IA
# app.py — Dashboard principal (Streamlit)
#
# Ponto de entrada do software. Orquestra os três módulos em um painel
# interativo com navegação multi-página e visual de pitch acadêmico.
#
# Para rodar localmente:
#   streamlit run app.py
#
# Para publicar:
#   Push para GitHub → Streamlit Community Cloud (streamlit.io/cloud)
# =============================================================================

import io
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Adicionar diretório raiz ao path
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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para visual de pitch
st.markdown("""
<style>
    /* Ocultar menu hamburger e footer padrão */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Header principal */
    .co-ia-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .co-ia-header h1 { color: white; margin: 0; font-size: 2rem; }
    .co-ia-header p  { color: #a8b2d8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
    }

    /* Badge de patente */
    .patent-badge {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #1E88E5;
        color: #1565C0;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
    }

    /* Zona de risco */
    .risk-zone-safe     { background:#e8f5e9; border-left:4px solid #43A047; padding:0.8rem; border-radius:8px; }
    .risk-zone-warning  { background:#fff8e1; border-left:4px solid #FB8C00; padding:0.8rem; border-radius:8px; }
    .risk-zone-critical { background:#ffebee; border-left:4px solid #EF5350; padding:0.8rem; border-radius:8px; }

    /* Separadores de seção */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar — navegação e upload
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.image("https://img.shields.io/badge/CO--IA-v1.0.0-1E88E5?style=for-the-badge", use_container_width=True)
        st.markdown("### 📂 Dados de Entrada")

        fonte = st.radio(
            "Fonte dos dados:",
            ["🎯 Dataset de Demonstração", "📤 Upload de CSV/JSONL"],
            index=0,
        )

        df_input = None
        if "dataset de demonstração" in fonte.lower():
            demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_dataset.csv")
            df_input = pd.read_csv(demo_path)
            st.success(f"✅ Demo carregada: **{len(df_input)}** registros")
            st.caption("12 textos humanos + 12 sintéticos rotulados.")
        else:
            arquivo = st.file_uploader(
                "Envie seu arquivo:",
                type=["csv", "jsonl"],
                help="CSV com coluna obrigatória 'texto'. Colunas opcionais: 'id', 'fonte', 'tipo_real'."
            )
            if arquivo:
                if arquivo.name.endswith(".jsonl"):
                    import json
                    linhas = [json.loads(l) for l in arquivo.read().decode().split("\n") if l.strip()]
                    df_input = pd.DataFrame(linhas)
                else:
                    df_input = pd.read_csv(arquivo)

                if "texto" not in df_input.columns:
                    st.error("❌ O arquivo deve conter a coluna **texto**.")
                    df_input = None
                else:
                    st.success(f"✅ Carregado: **{len(df_input)}** registros")
            else:
                st.info("Aguardando upload…")

        st.markdown("---")
        st.markdown("### ⚙️ Configurações")
        limiar = st.slider(
            "Limiar de classificação:",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            help="Score ≥ limiar → sintético. Ref: IN202511107978-A"
        )
        n_geracoes = st.slider(
            "Gerações na simulação:",
            min_value=3,
            max_value=15,
            value=8,
            help="US2025094459-A1 (Madisetti)"
        )

        st.markdown("---")
        st.caption(f"**{APP_TITLE}** {APP_VERSION}  \n{APP_AUTHOR}")

    return df_input, limiar, n_geracoes


# =============================================================================
# Aba 0 — Visão Geral / Home
# =============================================================================

def render_home(df: pd.DataFrame):
    st.markdown("""
    <div class="co-ia-header">
        <h1>🔬 CO-IA — Curadoria de Origem para IA</h1>
        <p>Pipeline integrado de detecção, monitoramento e proveniência para mitigação do Model Collapse</p>
        <p style="font-size:0.85rem;color:#7986cb;margin-top:0.5rem;">
            UNIRIO / BSI · Metodologia Científica e Tecnológica · 2025 |
            Fundamentado em 71 patentes (Derwent + INPI) e 244 estudos científicos
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Módulo I — Filtro de Entrada**\n\nCalcula *contamination score* (0–1) e classifica cada texto como humano ou sintético usando 5 estratégias estatísticas explicáveis.")
    with col2:
        st.warning("**Módulo II — Monitoramento**\n\nCalcula métricas de diversidade do corpus, exibe o risco de colapso e simula a degradação ao longo de N gerações de treinamento recursivo.")
    with col3:
        st.success("**Módulo III — Proveniência**\n\nGera hash SHA-256 por registro, protege o padrão-ouro e recomenda o reequilíbrio humano:sintético do corpus.")

    st.markdown('<div class="section-title">📊 Visão Rápida do Corpus Carregado</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total de Registros", len(df))
    if "tipo_real" in df.columns:
        n_hum = (df["tipo_real"] == "humano").sum()
        n_sin = (df["tipo_real"] == "sintetico").sum()
        col_b.metric("Humanos (rótulo real)", n_hum)
        col_c.metric("Sintéticos (rótulo real)", n_sin)
        col_d.metric("Colunas disponíveis", len(df.columns))
    else:
        col_b.metric("Colunas", len(df.columns))

    st.markdown("**Prévia dos dados:**")
    st.dataframe(df.head(5), use_container_width=True)

    # Mapa de patentes
    st.markdown('<div class="section-title">📜 Fundamentação — Patentes por Módulo</div>', unsafe_allow_html=True)
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.markdown("**🔴 Módulo I — Filtro**")
        for p in PATENT_REFERENCES["modulo1"]:
            st.markdown(f'<span class="patent-badge">{p["id"]}</span> {p["tecnica"]}', unsafe_allow_html=True)
    with pcol2:
        st.markdown("**🟠 Módulo II — Monitor**")
        for p in PATENT_REFERENCES["modulo2"]:
            st.markdown(f'<span class="patent-badge">{p["id"]}</span> {p["tecnica"]}', unsafe_allow_html=True)
    with pcol3:
        st.markdown("**🟢 Módulo III — Proveniência**")
        for p in PATENT_REFERENCES["modulo3"]:
            st.markdown(f'<span class="patent-badge">{p["id"]}</span> {p["tecnica"]}', unsafe_allow_html=True)


# =============================================================================
# Aba 1 — Módulo I: Filtro de Entrada
# =============================================================================

def render_modulo1(df: pd.DataFrame, limiar: float):
    st.markdown('<div class="section-title">🔴 Módulo I — Filtro de Entrada (Contamination Score)</div>', unsafe_allow_html=True)
    st.caption("Ref: WO2025037142-A1 · IN202511107978-A · CN119358696-A · US2024354648-A1")

    engine = build_default_engine(limiar=limiar)

    with st.spinner("Calculando contamination scores…"):
        df_resultado = engine.analisar_dataset(df)
        metricas = engine.metricas_resumo(df_resultado)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analisado", metricas["total_registros"])
    c2.metric("✅ Humanos", metricas["n_humanos"],
              delta=f"-{metricas['taxa_contaminacao']}% contaminação" if metricas['taxa_contaminacao'] > 0 else None)
    c3.metric("⚠️ Sintéticos", metricas["n_sinteticos"])
    c4.metric("Score Médio", metricas["score_medio"],
              delta="limiar: " + str(limiar), delta_color="off")

    # Gráfico de distribuição de scores
    st.markdown("#### Distribuição do Contamination Score")
    fig_hist = px.histogram(
        df_resultado,
        x="score_contaminacao",
        color="classificacao_coia",
        nbins=20,
        color_discrete_map={"humano": COLOR_HUMANO, "sintético": COLOR_SINTETICO},
        labels={"score_contaminacao": "Contamination Score", "classificacao_coia": "Classificação CO-IA"},
        title="Histograma de Scores — Módulo I",
    )
    fig_hist.add_vline(x=limiar, line_dash="dash", line_color="#1E88E5",
                       annotation_text=f"Limiar ({limiar})", annotation_position="top right")
    fig_hist.update_layout(height=350, margin=dict(t=50, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Gráfico radar de scores por estratégia (média por classificação)
    st.markdown("#### Scores por Estratégia (média por classe)")
    score_cols = [c for c in df_resultado.columns if c.startswith("score_") and c != "score_contaminacao"]
    if score_cols:
        medias = df_resultado.groupby("classificacao_coia")[score_cols].mean().reset_index()
        estrategias = [c.replace("score_", "").replace("_", " ").title() for c in score_cols]

        fig_radar = go.Figure()
        cores = {"humano": COLOR_HUMANO, "sintético": COLOR_SINTETICO}
        for _, row in medias.iterrows():
            valores = [row[c] for c in score_cols]
            valores.append(valores[0])  # fechar radar
            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=estrategias + [estrategias[0]],
                fill="toself",
                name=row["classificacao_coia"],
                line_color=cores.get(row["classificacao_coia"], COLOR_NEUTRO),
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=380, title="Perfil de Scores por Estratégia",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tabela detalhada
    st.markdown("#### Tabela de Resultados")
    colunas_exibir = ["id", "texto", "score_contaminacao", "classificacao_coia"] + score_cols[:3]
    colunas_exibir = [c for c in colunas_exibir if c in df_resultado.columns]

    # Formatação condicional
    def colorir_linha(row):
        cor = "#ffeaea" if row["classificacao_coia"] == "sintético" else "#eafff0"
        return [f"background-color: {cor}"] * len(row)

    st.dataframe(
        df_resultado[colunas_exibir].style.apply(colorir_linha, axis=1),
        use_container_width=True,
        height=400,
    )

    # Avaliação de acurácia se rótulo real disponível
    if "tipo_real" in df_resultado.columns:
        st.markdown("#### Avaliação de Acurácia (comparação com rótulo real)")
        df_resultado["correto"] = df_resultado.apply(
            lambda r: r["classificacao_coia"] == r["tipo_real"] or
                      (r["classificacao_coia"] == "sintético" and r["tipo_real"] == "sintetico"),
            axis=1
        )
        acuracia = df_resultado["correto"].mean() * 100
        st.metric("Acurácia do CO-IA vs rótulo real", f"{acuracia:.1f}%")

    # Salvar resultado no estado de sessão para uso nos módulos seguintes
    st.session_state["df_m1"] = df_resultado
    st.session_state["metricas_m1"] = metricas

    return df_resultado


# =============================================================================
# Aba 2 — Módulo II: Monitoramento de Degeneração
# =============================================================================

def render_modulo2(df: pd.DataFrame, n_geracoes: int):
    st.markdown('<div class="section-title">🟠 Módulo II — Monitoramento de Degeneração</div>', unsafe_allow_html=True)
    st.caption("Ref: US2025342187-A1 · US2025094459-A1 · US2025217394-A1 · US2024411789-A1 (Madisetti V.)")

    textos = df["texto"].tolist() if "texto" in df.columns else []
    df_m1 = st.session_state.get("df_m1", df)

    with st.spinner("Calculando métricas de diversidade e risco…"):
        resultado = analisar_corpus(textos, df=df_m1, n_geracoes=n_geracoes)

    metricas = resultado["metricas"]
    risco    = resultado["risco"]
    sim      = resultado["simulacao"]

    # --- Gauge de risco ---
    st.markdown("#### Índice de Risco de Colapso")
    risco_val = risco["risco_total"]
    cor_gauge = COLOR_HUMANO if risco_val < COLLAPSE_RISK_THRESHOLDS["seguro"] else \
                COLOR_ALERTA if risco_val < COLLAPSE_RISK_THRESHOLDS["atencao"] else \
                COLOR_SINTETICO

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risco_val * 100,
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": f"Risco de Colapso de Modelo<br><sub>{risco['zona']}</sub>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": cor_gauge},
            "steps": [
                {"range": [0,  COLLAPSE_RISK_THRESHOLDS["seguro"]  * 100], "color": "#e8f5e9"},
                {"range": [COLLAPSE_RISK_THRESHOLDS["seguro"]  * 100,
                           COLLAPSE_RISK_THRESHOLDS["atencao"] * 100], "color": "#fff8e1"},
                {"range": [COLLAPSE_RISK_THRESHOLDS["atencao"] * 100,
                           COLLAPSE_RISK_THRESHOLDS["critico"] * 100], "color": "#fff3e0"},
                {"range": [COLLAPSE_RISK_THRESHOLDS["critico"] * 100, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#1a1a2e", "width": 3},
                "thickness": 0.75,
                "value": risco_val * 100,
            },
        },
        delta={"reference": COLLAPSE_RISK_THRESHOLDS["seguro"] * 100,
               "decreasing": {"color": COLOR_HUMANO},
               "increasing": {"color": COLOR_SINTETICO}},
    ))
    fig_gauge.update_layout(height=320, margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Métricas detalhadas ---
    st.markdown("#### Métricas de Diversidade do Corpus")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Entropia (bigramas)", f"{metricas['entropia_bigramas']:.3f}", help="Shannon entropy. Saudável: > 5.0")
    mc2.metric("TTR do Corpus", f"{metricas['ttr']:.3f}", help="Type-Token Ratio. Saudável: > 0.35")
    mc3.metric("MATTR", f"{metricas['mattr']:.3f}", help="Moving-Average TTR. Métrica mais estável.")
    mc4.metric("Prop. Sintéticos", f"{metricas['proporcao_sinteticos']*100:.1f}%", help="% registros classificados sintéticos")

    # Componentes de risco
    comp = risco["componentes"]
    st.markdown("#### Decomposição do Risco")
    fig_bar = go.Figure(go.Bar(
        x=list(comp.values()),
        y=["Risco Entropia", "Risco TTR", "Risco MATTR", "Risco % Sintéticos"],
        orientation="h",
        marker_color=[COLOR_SINTETICO if v > 0.5 else COLOR_ALERTA if v > 0.3 else COLOR_HUMANO
                      for v in comp.values()],
        text=[f"{v:.2f}" for v in comp.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 1.1], title="Componente de Risco (0–1)"),
        height=250, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Simulação de gerações ---
    st.markdown("#### Simulação de Gerações de Treinamento Recursivo")
    st.caption("Baseado em US2025342187-A1 e US2025094459-A1 (Madisetti) · Shumailov et al. (2024)")

    df_sim = sim
    tab_entr, tab_ttr, tab_risco = st.tabs(["Entropia", "TTR / MATTR", "Risco de Colapso"])

    with tab_entr:
        fig_e = px.line(
            df_sim, x="geracao", y="entropia_bigramas", color="cenario",
            color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
            title="Entropia de Bigramas ao Longo das Gerações",
            labels={"geracao": "Geração", "entropia_bigramas": "Entropia"},
            markers=True,
        )
        fig_e.add_hline(y=5.0, line_dash="dot", line_color=COLOR_NEUTRO,
                        annotation_text="Mínimo saudável (5.0)")
        st.plotly_chart(fig_e, use_container_width=True)

    with tab_ttr:
        fig_t = px.line(
            df_sim, x="geracao", y="ttr", color="cenario",
            color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
            title="TTR ao Longo das Gerações",
            labels={"geracao": "Geração", "ttr": "Type-Token Ratio"},
            markers=True,
        )
        fig_t.add_hline(y=0.35, line_dash="dot", line_color=COLOR_NEUTRO,
                        annotation_text="Mínimo saudável (0.35)")
        st.plotly_chart(fig_t, use_container_width=True)

    with tab_risco:
        fig_r = px.line(
            df_sim, x="geracao", y="risco_colapso", color="cenario",
            color_discrete_map={"Com CO-IA": COLOR_HUMANO, "Sem CO-IA": COLOR_SINTETICO},
            title="Evolução do Risco de Colapso",
            labels={"geracao": "Geração", "risco_colapso": "Índice de Risco"},
            markers=True,
        )
        for limiar_nome, limiar_val in COLLAPSE_RISK_THRESHOLDS.items():
            fig_r.add_hline(y=limiar_val, line_dash="dash", line_color=COLOR_ALERTA,
                            annotation_text=limiar_nome)
        st.plotly_chart(fig_r, use_container_width=True)

    # Salvar para Módulo III
    st.session_state["metricas_m2"] = metricas
    st.session_state["risco_m2"] = risco

    return metricas, risco


# =============================================================================
# Aba 3 — Módulo III: Auditoria de Proveniência
# =============================================================================

def render_modulo3(df: pd.DataFrame):
    st.markdown('<div class="section-title">🟢 Módulo III — Auditoria de Proveniência</div>', unsafe_allow_html=True)
    st.caption("Ref: US2025238634-A1 · US2026080037-A1 · BR 11 2026 0105 (G06F 21/16) · BR 11 2023 0065 (G06N 5/02)")

    df_m1 = st.session_state.get("df_m1", df)
    if "classificacao_coia" not in df_m1.columns:
        engine = build_default_engine()
        df_m1 = engine.analisar_dataset(df_m1)

    # Seleção do padrão-ouro
    st.markdown("#### Definir Padrão-Ouro (Gold Standard)")
    st.caption("Marque os registros de origem humana verificada para protegê-los contra Data Poisoning. Ref: US2025238634-A1")

    candidatos_gold = df_m1[df_m1["classificacao_coia"] == "humano"]["id"].tolist() \
        if "id" in df_m1.columns else []

    ids_gold = st.multiselect(
        "Selecionar IDs para padrão-ouro:",
        options=candidatos_gold,
        default=candidatos_gold[:min(4, len(candidatos_gold))],
        help="Esses registros serão protegidos e priorizados no reequilíbrio."
    )

    with st.spinner("Gerando registros de proveniência (SHA-256)…"):
        resultado_prov = gerar_proveniencia(df_m1, ids_gold=ids_gold)

    reequilibrio = resultado_prov["reequilibrio"]
    df_prov      = resultado_prov["df_provenance"]

    # --- Reequilíbrio ---
    st.markdown("#### Recomendação de Reequilíbrio do Corpus")
    eq_cols = st.columns(4)
    eq_cols[0].metric("Humanos", f"{reequilibrio['proporcao_humanos']}%", f"{reequilibrio['n_humanos']} reg.")
    eq_cols[1].metric("Sintéticos", f"{reequilibrio['proporcao_sinteticos']}%", f"{reequilibrio['n_sinteticos']} reg.")
    eq_cols[2].metric("Padrão-Ouro", f"{reequilibrio['proporcao_gold']}%", f"{reequilibrio['n_gold_standard']} reg.")
    eq_cols[3].metric("Limite Sintéticos", f"≤ {reequilibrio['limite_sinteticos']:.0f}%", "recomendado")

    # Gráfico de pizza de composição
    fig_pie = px.pie(
        values=[reequilibrio["n_humanos"], reequilibrio["n_sinteticos"]],
        names=["Humanos", "Sintéticos"],
        color_discrete_sequence=[COLOR_HUMANO, COLOR_SINTETICO],
        title="Composição Atual do Corpus",
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Status e ação
    status = reequilibrio["status"]
    if "✅" in status:
        st.success(f"**{status}**  \n{reequilibrio['acao_recomendada']}")
    elif "⚠️" in status:
        st.error(f"**{status}**  \n{reequilibrio['acao_recomendada']}")
    else:
        st.warning(f"**{status}**  \n{reequilibrio['acao_recomendada']}")

    # Tabela de proveniência
    st.markdown("#### Registros de Proveniência (SHA-256)")
    if not df_prov.empty:
        df_exib = df_prov.copy()
        if "hash_sha256" in df_exib.columns:
            df_exib["hash_sha256"] = df_exib["hash_sha256"].str[:20] + "…"
        colunas = ["record_id", "hash_sha256", "fonte", "tipo_declarado",
                   "score_contaminacao", "is_gold_standard", "comprimento_chars"]
        colunas_ok = [c for c in colunas if c in df_exib.columns]
        st.dataframe(df_exib[colunas_ok], use_container_width=True, height=350)

    # Salvar para exportação
    st.session_state["reequilibrio"] = reequilibrio
    st.session_state["df_prov"] = df_prov

    return reequilibrio, df_prov


# =============================================================================
# Aba 4 — Exportação do Relatório
# =============================================================================

def render_exportacao():
    st.markdown('<div class="section-title">📄 Exportar Relatório de Auditoria</div>', unsafe_allow_html=True)
    st.caption("Relatório em Markdown exportável — evidência para pitch acadêmico e registro INPI/DIT")

    metricas_m1  = st.session_state.get("metricas_m1",  {"total_registros": 0, "n_humanos": 0, "n_sinteticos": 0, "taxa_contaminacao": 0.0, "score_medio": 0.0})
    risco_m2     = st.session_state.get("risco_m2",     {"zona": "N/A", "risco_total": 0.0})
    reequilibrio = st.session_state.get("reequilibrio", {"status": "Execute os módulos primeiro."})
    df_prov      = st.session_state.get("df_prov",      pd.DataFrame())

    builder = (
        ReportBuilder()
        .adicionar_cabecalho(APP_TITLE, APP_SUBTITLE, APP_VERSION)
        .adicionar_resumo_execucao(metricas_m1, risco_m2, reequilibrio)
        .adicionar_registros_proveniencia(df_prov)
        .adicionar_reequilibrio(reequilibrio)
        .adicionar_fundamentacao()
        .adicionar_rodape()
    )
    relatorio_md = builder.build()

    st.markdown("#### Prévia do Relatório")
    with st.expander("Ver conteúdo Markdown", expanded=True):
        st.markdown(relatorio_md)

    # Download
    st.download_button(
        label="⬇️ Baixar Relatório (.md)",
        data=relatorio_md.encode("utf-8"),
        file_name="relatorio_co_ia.md",
        mime="text/markdown",
    )

    # Resumo para registro INPI/DIT
    st.markdown("#### Resumo Técnico — Registro de Software (INPI/DIT)")
    resumo_inpi = f"""
**Nome do Programa:** CO-IA — Curadoria de Origem para IA
**Versão:** {APP_VERSION}
**Linguagem de Programação:** Python 3.11
**Plataforma:** Streamlit (web) — CPU only, sem GPU

**Finalidade:**
Software para mitigação do Colapso de Modelos (Model Collapse) em sistemas de
Inteligência Artificial Generativa, por meio da curadoria integrada de origem dos
dados de treinamento. Combina detecção estatística de dados sintéticos (Módulo I),
monitoramento de diversidade e risco do corpus (Módulo II) e auditoria de
proveniência com rastreamento criptográfico (Módulo III).

**Módulos:**
1. Filtro de Entrada — ensemble de 5 estratégias de pontuação (contamination score 0–1)
2. Monitoramento de Degeneração — métricas Shannon/TTR/MATTR + simulação de gerações
3. Auditoria de Proveniência — hashing SHA-256, padrão-ouro, reequilíbrio de corpus

**Embasamento técnico:** 71 patentes (Derwent Innovation + INPI) e 244 estudos
científicos mapeados sistematicamente no âmbito da disciplina MCT — UNIRIO / BSI.

**Autor:** Gabriel de Melo Guedes Souza
**Instituição:** UNIRIO — Bacharelado em Sistemas de Informação (BSI)
"""
    with st.expander("Ver resumo INPI/DIT"):
        st.markdown(resumo_inpi)

    st.download_button(
        label="⬇️ Baixar Resumo INPI (.txt)",
        data=resumo_inpi.encode("utf-8"),
        file_name="resumo_inpi_co_ia.txt",
        mime="text/plain",
    )


# =============================================================================
# Main — orquestrador de páginas
# =============================================================================

def main():
    df_input, limiar, n_geracoes = render_sidebar()

    if df_input is None:
        st.markdown("""
        <div class="co-ia-header">
            <h1>🔬 CO-IA — Curadoria de Origem para IA</h1>
            <p>Carregue um dataset na barra lateral ou use o dataset de demonstração para começar.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Selecione 'Dataset de Demonstração' na barra lateral para ver o sistema em ação.")
        return

    # Garantir coluna 'id'
    if "id" not in df_input.columns:
        df_input["id"] = [f"REC_{i:04d}" for i in range(len(df_input))]

    # Tabs de navegação
    abas = st.tabs([
        "🏠 Visão Geral",
        "🔴 Módulo I — Filtro",
        "🟠 Módulo II — Monitor",
        "🟢 Módulo III — Proveniência",
        "📄 Exportar Relatório",
    ])

    with abas[0]:
        render_home(df_input)

    with abas[1]:
        df_m1 = render_modulo1(df_input, limiar)

    with abas[2]:
        df_para_m2 = st.session_state.get("df_m1", df_input)
        render_modulo2(df_para_m2, n_geracoes)

    with abas[3]:
        render_modulo3(df_input)

    with abas[4]:
        render_exportacao()


if __name__ == "__main__":
    main()
