# =============================================================================
# PROVYN — Curadoria de Origem para IA
# modules/modulo1_filtro.py — Modulo I: Filtro de Entrada
#
# Padroes de projeto: Strategy (ScoreStrategy), Facade (FilterEngine),
#                     Factory (build_default_engine)
#
# Patentes de referencia:
#   WO2025037142-A1  — NEC Lab — curadoria e qualidade de dados sinteticos
#   IN202511107978-A — Univ. Manipal — deteccao ML de dados sinteticos
#   US2024354648-A1  — filtragem anomala de corpus (G06N 020/00)
#   CN119358696-A    — ensemble de features para filtragem (G06F 018/21)
# =============================================================================

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CONTAMINATION_WEIGHTS, DEFAULT_THRESHOLD

# =============================================================================
# Strategy — interface
# =============================================================================


class ScoreStrategy(ABC):
    """Interface para estrategias de pontuacao de contaminacao."""

    @abstractmethod
    def calcular(self, texto: str) -> float:
        """Retorna score 0.0-1.0 (1.0 = alta probabilidade de sintetico)."""
        ...


# =============================================================================
# Estrategias concretas
# =============================================================================


class MarcadoresLLMStrategy(ScoreStrategy):
    """Detecta marcadores lexicos tipicos de LLMs.

    Ref: IN202511107978-A (Univ. Manipal) — feature engineering para deteccao.
    """

    _PADROES = [
        r"\bem suma\b",
        r"\bpor outro lado\b",
        r"\baltamente\b",
        r"\be importante (ressaltar|notar|destacar|mencionar)\b",
        r"\be fundamental\b",
        r"\be essencial\b",
        r"\be crucial\b",
        r"\bcabe destacar\b",
        r"\bcabe ressaltar\b",
        r"\bno contexto (de|do|da)\b",
        r"\bno ambito\b",
        r"\bpode-se (afirmar|concluir|observar|dizer)\b",
        r"\be (valido|necessario|relevante|pertinente)\b",
        r"\bdessa forma\b",
        r"\bnesse sentido\b",
        r"\bin summary\b",
        r"\bin conclusion\b",
        r"\bit is (important|crucial|essential|worth) (to note|to mention|noting)\b",
        r"\bon the other hand\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bnevertheless\b",
        r"\bin this context\b",
        r"\bultimately\b",
        r"\boverall\b",
        r"\bin terms of\b",
    ]
    _REGEX = [re.compile(p, re.IGNORECASE) for p in _PADROES]

    def calcular(self, texto: str) -> float:
        if not texto or not texto.strip():
            return 0.5
        hits = sum(1 for rx in self._REGEX if rx.search(texto))
        return round(min(hits / 3.0, 1.0), 4)


class UniformidadeSentencasStrategy(ScoreStrategy):
    """Mede a uniformidade de comprimento das sentencas.

    LLMs tendem a gerar sentencas com comprimento mais uniforme que humanos.
    Ref: WO2025037142-A1 (NEC Lab).
    """

    def calcular(self, texto: str) -> float:
        if not texto or not texto.strip():
            return 0.5
        sentencas = re.split(r"[.!?]+", texto)
        sentencas = [s.strip() for s in sentencas if len(s.strip()) > 10]
        if len(sentencas) < 2:
            return 0.3
        comprimentos = [len(s.split()) for s in sentencas]
        media = np.mean(comprimentos)
        if media == 0:
            return 0.3
        cv = np.std(comprimentos) / media
        score = max(0.0, 1.0 - cv / 0.5)
        return round(min(score, 1.0), 4)


class ComprimentoPalavrasStrategy(ScoreStrategy):
    """Mede o comprimento medio das palavras.

    LLMs tendem a usar vocabulario formal com palavras mais longas.
    Ref: CN119358696-A (G06F 018/21).
    """

    def calcular(self, texto: str) -> float:
        if not texto or not texto.strip():
            return 0.5
        palavras = re.findall(r"\b[a-zA-Z]+\b", texto)
        if not palavras:
            return 0.5
        media = np.mean([len(p) for p in palavras])
        score = (media - 4.5) / (8.0 - 4.5)
        return round(min(max(score, 0.0), 1.0), 4)


class EstruturaParagrafoStrategy(ScoreStrategy):
    """Detecta padroes estruturais tipicos de LLMs em paragrafos.

    Ref: US2024354648-A1 (G06N 020/00).
    """

    _INICIOS = re.compile(
        r"^(Primeiramente|Em primeiro lugar|Inicialmente|Alem disso|"
        r"Por fim|Finalmente|Em suma|Dessa forma|Nesse sentido|"
        r"First(ly)?|Secondly|Furthermore|Moreover|In addition|Finally|"
        r"In conclusion|To summarize|Overall)",
        re.IGNORECASE | re.MULTILINE,
    )
    _CONCLUSAO = re.compile(
        r"(em conclusao|concluindo|portanto|assim sendo|"
        r"in conclusion|to conclude|therefore|in summary)",
        re.IGNORECASE,
    )

    def calcular(self, texto: str) -> float:
        if not texto or not texto.strip():
            return 0.5
        paragrafos = [p.strip() for p in texto.split("\n") if len(p.strip()) > 20]
        if not paragrafos:
            score = 0.1
            if self._INICIOS.search(texto):
                score += 0.4
            if self._CONCLUSAO.search(texto):
                score += 0.3
            return round(min(score, 1.0), 4)
        hits_inicio = sum(1 for p in paragrafos if self._INICIOS.match(p))
        hits_conclusao = 1 if self._CONCLUSAO.search(texto) else 0
        proporcao_inicio = hits_inicio / len(paragrafos)
        score = proporcao_inicio * 0.7 + hits_conclusao * 0.3
        return round(min(score, 1.0), 4)


class EntropiaBigramasStrategy(ScoreStrategy):
    """Mede a diversidade de bigramas de palavras via entropia de Shannon.

    LLMs geram bigramas mais repetitivos que textos humanos.
    Ref: US2025094459-A1 (Madisetti) — metricas de diversidade.
    """

    def calcular(self, texto: str) -> float:
        if not texto or not texto.strip():
            return 0.5
        palavras = re.findall(r"\b\w+\b", texto.lower())
        if len(palavras) < 4:
            return 0.3
        bigramas = [(palavras[i], palavras[i + 1]) for i in range(len(palavras) - 1)]
        freq: Dict[tuple, int] = {}
        for bg in bigramas:
            freq[bg] = freq.get(bg, 0) + 1
        total = len(bigramas)
        entropia = -sum((c / total) * math.log2(c / total) for c in freq.values())
        entropia_max = math.log2(total) if total > 1 else 1.0
        diversidade = entropia / entropia_max if entropia_max > 0 else 0.5
        score = 1.0 - diversidade
        return round(min(max(score, 0.0), 1.0), 4)


# =============================================================================
# Facade — FilterEngine
# =============================================================================


class FilterEngine:
    """Orquestra as estrategias de pontuacao e classifica textos.

    Padrao Facade: expoe API simples sobre o ensemble de estrategias.
    Ref: WO2025037142-A1, IN202511107978-A.
    """

    def __init__(
        self,
        estrategias: List[ScoreStrategy] | None = None,
        pesos: Dict[str, float] | None = None,
        limiar: float = DEFAULT_THRESHOLD,
    ):
        self.limiar = limiar
        self.estrategias = estrategias or _estrategias_padrao()
        pesos_raw = pesos or CONTAMINATION_WEIGHTS
        self._chaves_pesos = list(pesos_raw.keys())
        total = sum(pesos_raw.values()) or 1.0
        self._pesos_norm = {k: v / total for k, v in pesos_raw.items()}

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def analisar_texto(self, texto: str) -> Dict:
        """Analisa um unico texto e retorna scores detalhados."""
        scores_individuais = {}
        for estrategia, chave in zip(self.estrategias, self._chaves_pesos):
            try:
                scores_individuais[chave] = round(estrategia.calcular(texto), 4)
            except Exception:
                scores_individuais[chave] = 0.3

        score_total = sum(
            scores_individuais[k] * self._pesos_norm.get(k, 0)
            for k in self._chaves_pesos
        )
        score_total = round(min(max(score_total, 0.0), 1.0), 4)
        classificacao = "sintetico" if score_total >= self.limiar else "humano"

        return {
            "score_contaminacao": score_total,
            "classificacao": classificacao,
            "scores_detalhados": scores_individuais,
        }

    def analisar_dataset(self, df: pd.DataFrame, col_texto: str = "texto") -> pd.DataFrame:
        """Analisa DataFrame inteiro e retorna enriquecido com scores."""
        resultados = df[col_texto].apply(self.analisar_texto)

        df = df.copy()
        df["score_contaminacao"] = resultados.apply(lambda r: r["score_contaminacao"])
        df["classificacao_provyn"] = resultados.apply(lambda r: r["classificacao"])

        for chave in self._chaves_pesos:
            df[f"score_{chave}"] = resultados.apply(
                lambda r, k=chave: r["scores_detalhados"][k]
            )
        return df

    def filtrar_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retorna (corpus_limpo, corpus_sintetico)."""
        if "classificacao_provyn" not in df.columns:
            df = self.analisar_dataset(df)
        limpo = df[df["classificacao_provyn"] == "humano"].copy()
        sintetico = df[df["classificacao_provyn"] == "sintetico"].copy()
        return limpo, sintetico

    def metricas_resumo(self, df: pd.DataFrame) -> Dict:
        """Retorna metricas agregadas do dataset ja analisado.

        Ref: CN119358696-A (G06F 018/21) — metricas de ensemble para auditoria.
        """
        if "score_contaminacao" not in df.columns:
            return {}
        total = len(df)
        n_sinteticos = int((df["classificacao_provyn"] == "sintetico").sum()) \
            if "classificacao_provyn" in df.columns else 0
        n_humanos = total - n_sinteticos
        return {
            "total_registros":   total,
            "n_humanos":         n_humanos,
            "n_sinteticos":      n_sinteticos,
            "taxa_contaminacao": round(n_sinteticos / total * 100, 1) if total > 0 else 0.0,
            "score_medio":       round(df["score_contaminacao"].mean(), 3),
            "score_max":         round(df["score_contaminacao"].max(), 3),
            "score_min":         round(df["score_contaminacao"].min(), 3),
        }


# =============================================================================
# Factory
# =============================================================================


def _estrategias_padrao() -> List[ScoreStrategy]:
    return [
        MarcadoresLLMStrategy(),
        UniformidadeSentencasStrategy(),
        ComprimentoPalavrasStrategy(),
        EstruturaParagrafoStrategy(),
        EntropiaBigramasStrategy(),
    ]


def build_default_engine(limiar: float = DEFAULT_THRESHOLD) -> FilterEngine:
    """Factory: cria FilterEngine com estrategias e pesos padrao."""
    return FilterEngine(limiar=limiar)
