# =============================================================================
# PROVYN — Curadoria de Origem para IA
# config.py — Configurações globais e constantes do framework
#
# Embasamento: WO2025037142-A1 (NEC Lab), IN202511107978-A (Manipal)
# US2024354648-A1 definem limiares e pesos para sistemas de curadoria.
# =============================================================================

# ---------------------------------------------------------------------------
# Módulo I — Filtro de Entrada
# ---------------------------------------------------------------------------

# Pesos do ensemble de estratégias de pontuação
# Ref: CN119358696-A (G06F 018/21) — ensemble de features para filtragem
CONTAMINATION_WEIGHTS = {
    "marcadores_llm":         0.35,  # Marcadores léxicos típicos de LLM
    "uniformidade_sent":      0.20,  # Uniformidade de comprimento de sentenças
    "comprimento_palavras":   0.18,  # Vocabulário formal (comprimento médio)
    "estrutura_paragrafo":    0.15,  # Estrutura de parágrafo típica de LLM
    "entropia_bigramas":      0.12,  # Diversidade de bigramas de palavras
}

# Limiar de classificação: score >= THRESHOLD → sintético
# Calibrado no dataset de demo: humanos max=0.18, sintéticos min=0.23
# Ref: IN202511107978-A
DEFAULT_THRESHOLD = 0.22

# Tamanho dos n-gramas para análise de repetitividade
NGRAM_SIZE_FILTER = 4

# ---------------------------------------------------------------------------
# Módulo II — Monitoramento de Degeneração
# ---------------------------------------------------------------------------

# Limiares de risco de colapso (0-1)
# Ref: US2025342187-A1 (Madisetti) — zonas de risco para feedback loops
COLLAPSE_RISK_THRESHOLDS = {
    "seguro":   0.30,
    "atencao":  0.55,
    "critico":  0.75,
}

# Número padrão de gerações na simulação
DEFAULT_GENERATIONS = 8

# Entropia mínima saudável de bigramas de palavras
# Ref: US2025094459-A1 (Madisetti)
ENTROPY_HEALTHY_MIN = 5.0

# TTR mínimo saudável
TTR_HEALTHY_MIN = 0.35

# ---------------------------------------------------------------------------
# Módulo III — Proveniência
# ---------------------------------------------------------------------------

# Proporção mínima de dados humanos verificados (padrão-ouro)
# Ref: WO2025037142-A1 (NEC Lab)
GOLD_STANDARD_MIN_RATIO = 0.30

# Proporção máxima de dados sintéticos recomendada
SYNTHETIC_MAX_RATIO = 0.50

# Algoritmo de hash
HASH_ALGORITHM = "sha256"

# ---------------------------------------------------------------------------
# Interface / Dashboard
# ---------------------------------------------------------------------------

APP_TITLE    = "PROVYN — Curadoria de Origem para IA"
APP_SUBTITLE = "Pipeline integrado de detecção, monitoramento e proveniência de dados de treinamento"
APP_VERSION  = "1.0.0-MVP"
APP_AUTHOR   = "Gabriel de Melo Guedes Souza — UNIRIO / BSI"

COLOR_SINTETICO = "#EF5350"
COLOR_HUMANO    = "#43A047"
COLOR_ALERTA    = "#FB8C00"
COLOR_INFO      = "#1E88E5"
COLOR_NEUTRO    = "#78909C"

# ---------------------------------------------------------------------------
# Patentes de referência por módulo
# ---------------------------------------------------------------------------
PATENT_REFERENCES = {
    "modulo1": [
        {"id": "WO2025037142-A1",  "titular": "NEC Lab",         "tecnica": "Curadoria e qualidade de dados sintéticos"},
        {"id": "IN202511107978-A", "titular": "Univ. Manipal",   "tecnica": "Detecção ML de dados sintéticos"},
        {"id": "US2024354648-A1",  "titular": "—",               "tecnica": "Filtragem anômala de corpus (G06N 020/00)"},
        {"id": "CN119358696-A",    "titular": "—",               "tecnica": "Ensemble de features para filtragem (G06F 018/21)"},
    ],
    "modulo2": [
        {"id": "US2025342187-A1",  "titular": "Madisetti V.",    "tecnica": "Sistema multi-nível com feedback loop"},
        {"id": "US2025094459-A1",  "titular": "Madisetti V.",    "tecnica": "Monitoramento de gerações recursivas"},
        {"id": "US2025217394-A1",  "titular": "Madisetti V.",    "tecnica": "Métricas de degeneração de modelo"},
        {"id": "US2024411789-A1",  "titular": "Madisetti V.",    "tecnica": "Refinamento por geração e baseline"},
    ],
    "modulo3": [
        {"id": "US2025238634-A1",  "titular": "—",               "tecnica": "Proteção de origem de dados"},
        {"id": "US2026080037-A1",  "titular": "—",               "tecnica": "Curadoria com proveniência verificável"},
        {"id": "BR 11 2026 0105", "titular": "—",               "tecnica": "Proteção de direitos autorais IA (G06F 21/16)"},
        {"id": "BR 11 2023 0065", "titular": "—",               "tecnica": "Rastreamento de dados compartilhados (G06N 5/02)"},
    ],
}
