# =============================================================================
# CO-IA — Módulo II: Monitoramento de Degeneração
# modulo2_monitor.py
#
# Calcula métricas de diversidade do corpus, estima o risco de colapso de modelo
# e simula a degradação ao longo de N gerações de treinamento recursivo.
#
# Fundamentação em patentes:
#   • US2025342187-A1  (Madisetti) — sistema multi-nível com feedback loop
#   • US2025094459-A1  (Madisetti) — geração sintética e monitoramento de gerações
#   • US2025217394-A1  (Madisetti) — métricas de degeneração de modelo
#   • US2024411789-A1  (Madisetti) — refinamento por geração e baseline de risco
#
# Padrões de projeto:
#   • Template Method — CorpusMetrics define algoritmo; subclasses podem estender
#   • Strategy        — cada métrica é intercambiável
#   • Pipeline        — dados fluem: corpus → métricas → risco → simulação
# =============================================================================

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    COLLAPSE_RISK_THRESHOLDS,
    DEFAULT_GENERATIONS,
    ENTROPY_HEALTHY_MIN,
    TTR_HEALTHY_MIN,
)


# =============================================================================
# CorpusMetrics — calcula todas as métricas de diversidade
# =============================================================================

class CorpusMetrics:
    """
    Calcula métricas de diversidade e integridade do corpus.
    Cada método pode ser usado de forma independente (Strategy) ou em conjunto.

    Fundamentação: Madisetti (US2025342187-A1, US2025094459-A1) define que a
    monitoração contínua do corpus via entropia e diversidade lexical é o
    mecanismo central de detecção precoce de colapso.
    """

    # ------------------------------------------------------------------
    # Tokenização utilitária
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenizar(texto: str) -> List[str]:
        texto = texto.lower().translate(str.maketrans("", "", string.punctuation))
        return [t for t in texto.split() if t]

    @staticmethod
    def _ngrams_palavras(tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    # ------------------------------------------------------------------
    # Métrica 1: Entropia de Shannon de Bigramas de Palavras
    # Ref: US2025342187-A1 — baseline de entropia para feedback loop
    # ------------------------------------------------------------------

    def entropia_corpus(self, textos: List[str], n: int = 2) -> float:
        """
        Calcula a entropia de Shannon dos n-gramas de palavras sobre todo o corpus.
        Valores baixos indicam baixa diversidade → risco de colapso.
        """
        todos_tokens: List[str] = []
        for t in textos:
            todos_tokens.extend(self._tokenizar(t))

        if len(todos_tokens) < n + 1:
            return 0.0

        ngrams = self._ngrams_palavras(todos_tokens, n)
        if not ngrams:
            return 0.0

        contagens = Counter(ngrams)
        total = len(ngrams)
        probs = [c / total for c in contagens.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    # ------------------------------------------------------------------
    # Métrica 2: Type-Token Ratio (TTR) do corpus
    # Ref: IN202511107978-A; US2024411789-A1 — diversidade lexical
    # ------------------------------------------------------------------

    def ttr_corpus(self, textos: List[str]) -> float:
        """
        Calcula o TTR (Type-Token Ratio) agregado do corpus.
        TTR = |vocabulário único| / |total de tokens|
        """
        todos_tokens: List[str] = []
        for t in textos:
            todos_tokens.extend(self._tokenizar(t))

        if not todos_tokens:
            return 0.0

        return len(set(todos_tokens)) / len(todos_tokens)

    # ------------------------------------------------------------------
    # Métrica 3: Riqueza de Vocabulário — Moving-Average TTR (MATTR)
    # Ref: US2025217394-A1 — métrica de degeneração estável
    # ------------------------------------------------------------------

    def mattr(self, textos: List[str], janela: int = 50) -> float:
        """
        Moving-Average Type-Token Ratio: mais estável que o TTR simples para
        corpora de tamanhos variados. Média de TTRs em janelas deslizantes.
        """
        todos_tokens: List[str] = []
        for t in textos:
            todos_tokens.extend(self._tokenizar(t))

        if len(todos_tokens) < janela:
            return self.ttr_corpus(textos)

        ttrs = []
        for i in range(len(todos_tokens) - janela + 1):
            janela_tokens = todos_tokens[i:i + janela]
            ttrs.append(len(set(janela_tokens)) / janela)

        return float(np.mean(ttrs))

    # ------------------------------------------------------------------
    # Métrica 4: Proporção de Dados Sintéticos no Corpus
    # Ref: WO2025037142-A1 — curadoria de proporção humano:sintético
    # ------------------------------------------------------------------

    def proporcao_sinteticos(self, df: pd.DataFrame) -> float:
        """Calcula proporção de registros classificados como sintéticos."""
        if "classificacao_coia" not in df.columns:
            return 0.0
        total = len(df)
        if total == 0:
            return 0.0
        n_sinteticos = (df["classificacao_coia"] == "sintético").sum()
        return n_sinteticos / total

    # ------------------------------------------------------------------
    # Cálculo consolidado de todas as métricas
    # ------------------------------------------------------------------

    def calcular_todas(
        self, textos: List[str], df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Retorna dicionário com todas as métricas calculadas."""
        return {
            "entropia_bigramas":  round(self.entropia_corpus(textos, n=2), 4),
            "entropia_unigramas": round(self.entropia_corpus(textos, n=1), 4),
            "ttr":                round(self.ttr_corpus(textos), 4),
            "mattr":              round(self.mattr(textos), 4),
            "n_textos":           len(textos),
            "n_tokens_total":     sum(len(self._tokenizar(t)) for t in textos),
            "proporcao_sinteticos": round(self.proporcao_sinteticos(df) if df is not None else 0.0, 4),
        }


# =============================================================================
# CollapseRiskEvaluator — estima o risco de colapso
# Ref: US2025342187-A1 — zonas de risco para feedback loops
# =============================================================================

class CollapseRiskEvaluator:
    """
    Combina as métricas do corpus em um índice único de risco de colapso (0–1).
    Implementa as zonas de risco definidas nas patentes de Madisetti.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or COLLAPSE_RISK_THRESHOLDS

    def calcular_risco(self, metricas: Dict[str, float]) -> Dict:
        """
        Retorna o índice de risco e a zona (seguro / atenção / crítico).
        Lógica baseada em US2025342187-A1 e US2025217394-A1.
        """
        entropia = metricas.get("entropia_bigramas", 5.0)
        ttr = metricas.get("ttr", 0.5)
        mattr = metricas.get("mattr", 0.5)
        prop_sint = metricas.get("proporcao_sinteticos", 0.0)

        # --- Componente entropia (ref: US2025342187-A1) ---
        # Entropia saudável mínima: ENTROPY_HEALTHY_MIN (config)
        risco_entropia = max(0.0, (ENTROPY_HEALTHY_MIN - entropia) / ENTROPY_HEALTHY_MIN)
        risco_entropia = min(risco_entropia, 1.0)

        # --- Componente TTR (ref: US2024411789-A1) ---
        risco_ttr = max(0.0, (TTR_HEALTHY_MIN - ttr) / TTR_HEALTHY_MIN)
        risco_ttr = min(risco_ttr, 1.0)

        # --- Componente MATTR ---
        risco_mattr = max(0.0, (TTR_HEALTHY_MIN - mattr) / TTR_HEALTHY_MIN)
        risco_mattr = min(risco_mattr, 1.0)

        # --- Componente proporção de sintéticos (ref: WO2025037142-A1) ---
        risco_prop = prop_sint  # já em [0, 1]

        # --- Score composto (média ponderada) ---
        risco_total = (
            0.35 * risco_entropia +
            0.20 * risco_ttr +
            0.15 * risco_mattr +
            0.30 * risco_prop
        )
        risco_total = round(min(max(risco_total, 0.0), 1.0), 4)

        # --- Zona de risco ---
        if risco_total < self.thresholds["seguro"]:
            zona = "🟢 Seguro"
        elif risco_total < self.thresholds["atencao"]:
            zona = "🟡 Atenção"
        elif risco_total < self.thresholds["critico"]:
            zona = "🟠 Alto Risco"
        else:
            zona = "🔴 Crítico"

        return {
            "risco_total": risco_total,
            "zona": zona,
            "componentes": {
                "risco_entropia": round(risco_entropia, 4),
                "risco_ttr":      round(risco_ttr, 4),
                "risco_mattr":    round(risco_mattr, 4),
                "risco_prop":     round(risco_prop, 4),
            },
        }


# =============================================================================
# GenerationSimulator — simula N gerações de treinamento recursivo
# Ref: US2025094459-A1 (Madisetti) — degradação por feedback loop
#      US2025342187-A1              — comparação com e sem curadoria (CO-IA)
# =============================================================================

class GenerationSimulator:
    """
    Simula como o corpus se degrada ao longo de N gerações de treinamento
    recursivo — com e sem o pipeline CO-IA ativo.

    A degradação sem CO-IA segue curvas empíricas documentadas em
    US2025094459-A1 (Madisetti) e na literatura de Model Collapse
    (Shumailov et al., 2024; Briesch et al., 2023).

    Com CO-IA ativo, a curadoria preserva as métricas acima dos limiares mínimos.
    """

    def __init__(self, metricas_iniciais: Dict[str, float]):
        self.metricas_iniciais = metricas_iniciais

    def simular(
        self,
        n_geracoes: int = DEFAULT_GENERATIONS,
        coia_ativo: bool = False,
    ) -> List[Dict]:
        """
        Retorna lista de dicionários com métricas por geração.
        Parâmetros de degradação calibrados a partir de:
          - US2025342187-A1 (Madisetti): taxa de degradação 12–18% por geração
          - Shumailov et al. (2024): colapso observado após 3–5 gerações sem curadoria
        """
        resultados = []
        evaluator = CollapseRiskEvaluator()

        entropia_atual = self.metricas_iniciais.get("entropia_bigramas", 7.0)
        ttr_atual      = self.metricas_iniciais.get("ttr", 0.55)
        mattr_atual    = self.metricas_iniciais.get("mattr", 0.50)
        prop_sint      = self.metricas_iniciais.get("proporcao_sinteticos", 0.30)

        for gen in range(n_geracoes + 1):
            metricas_gen = {
                "entropia_bigramas": round(entropia_atual, 4),
                "ttr":               round(ttr_atual, 4),
                "mattr":             round(mattr_atual, 4),
                "proporcao_sinteticos": round(prop_sint, 4),
            }
            risco_info = evaluator.calcular_risco(metricas_gen)

            resultados.append({
                "geracao":           gen,
                "entropia_bigramas": metricas_gen["entropia_bigramas"],
                "ttr":               metricas_gen["ttr"],
                "mattr":             metricas_gen["mattr"],
                "prop_sinteticos":   metricas_gen["proporcao_sinteticos"],
                "risco_colapso":     risco_info["risco_total"],
                "zona":              risco_info["zona"],
                "coia_ativo":        coia_ativo,
            })

            if gen == n_geracoes:
                break

            # ── Aplicar degradação para próxima geração ──────────────────
            if not coia_ativo:
                # SEM CO-IA: degradação agressiva (US2025094459-A1)
                fator_ruido = random.uniform(0.02, 0.05)
                entropia_atual  *= (1 - random.uniform(0.08, 0.15))
                ttr_atual       *= (1 - random.uniform(0.06, 0.12))
                mattr_atual     *= (1 - random.uniform(0.05, 0.10))
                prop_sint        = min(prop_sint + random.uniform(0.05, 0.12), 0.98)
            else:
                # COM CO-IA: degradação mitigada (CO-IA preserva diversidade)
                # Leve queda natural, mas curadoria mantém acima dos mínimos
                entropia_atual  *= (1 - random.uniform(0.01, 0.03))
                ttr_atual       *= (1 - random.uniform(0.01, 0.02))
                mattr_atual     *= (1 - random.uniform(0.01, 0.02))
                prop_sint        = max(
                    prop_sint + random.uniform(-0.02, 0.03),
                    0.0
                )
                # Garantir mínimos saudáveis (ação do CO-IA)
                entropia_atual  = max(entropia_atual, ENTROPY_HEALTHY_MIN * 0.90)
                ttr_atual       = max(ttr_atual, TTR_HEALTHY_MIN * 0.90)
                mattr_atual     = max(mattr_atual, TTR_HEALTHY_MIN * 0.85)
                prop_sint        = min(prop_sint, 0.45)

            # Garantir limites físicos
            entropia_atual = max(entropia_atual, 0.1)
            ttr_atual      = max(min(ttr_atual, 1.0), 0.0)
            mattr_atual    = max(min(mattr_atual, 1.0), 0.0)
            prop_sint      = max(min(prop_sint, 1.0), 0.0)

        return resultados

    def comparar_cenarios(self, n_geracoes: int = DEFAULT_GENERATIONS) -> pd.DataFrame:
        """
        Roda a simulação para os dois cenários e retorna DataFrame comparativo.
        """
        random.seed(42)  # reproducibilidade para a demo
        sem_coia = self.simular(n_geracoes, coia_ativo=False)
        random.seed(42)
        com_coia = self.simular(n_geracoes, coia_ativo=True)

        df_sem = pd.DataFrame(sem_coia)
        df_com = pd.DataFrame(com_coia)

        df_sem["cenario"] = "Sem CO-IA"
        df_com["cenario"] = "Com CO-IA"

        return pd.concat([df_sem, df_com], ignore_index=True)


# =============================================================================
# MonitoringPipeline — Fachada pública do Módulo II
# =============================================================================

def analisar_corpus(
    textos: List[str],
    df: Optional[pd.DataFrame] = None,
    n_geracoes: int = DEFAULT_GENERATIONS,
) -> Dict:
    """
    Ponto de entrada do Módulo II.
    Retorna métricas, risco de colapso e simulação de gerações.
    """
    calc = CorpusMetrics()
    evaluator = CollapseRiskEvaluator()

    metricas = calc.calcular_todas(textos, df)
    risco = evaluator.calcular_risco(metricas)
    simulador = GenerationSimulator(metricas)
    simulacao = simulador.comparar_cenarios(n_geracoes)

    return {
        "metricas": metricas,
        "risco": risco,
        "simulacao": simulacao,
    }
