# =============================================================================
# CO-IA — Módulo I: Filtro de Entrada (Ingestão)
# modulo1_filtro.py
#
# Calcula o "contamination score" de cada registro textual e classifica como
# humano ou sintético. Utiliza um ensemble de estratégias de pontuação
# implementado com o padrão Strategy (GoF).
#
# Fundamentação em patentes:
#   • WO2025037142-A1  (NEC Lab)        — curadoria de qualidade de dados sintéticos
#   • IN202511107978-A (Univ. Manipal)  — detecção ML de dados sintéticos
#   • CN119358696-A    (G06F 018/21)    — ensemble de features para filtragem
#   • US2024354648-A1  (G06N 020/00)    — filtragem anômala de corpus de treino
#
# Padrões de projeto:
#   • Strategy  — cada dimensão de análise é uma ScoreStrategy intercambiável
#   • Facade    — FilterEngine expõe interface simples ao dashboard
#   • Factory   — build_default_engine() instancia o pipeline padrão
# =============================================================================

from __future__ import annotations

import math
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    CONTAMINATION_WEIGHTS,
    DEFAULT_THRESHOLD,
    NGRAM_SIZE_FILTER,
)


# =============================================================================
# Interface Strategy
# =============================================================================

class ScoreStrategy(ABC):
    """
    Interface base para todas as estratégias de pontuação.
    Retorna float em [0, 1]: 0 = provavelmente humano, 1 = provavelmente sintético.
    """

    @property
    @abstractmethod
    def nome(self) -> str:
        """Nome legível da estratégia."""

    @abstractmethod
    def calcular(self, texto: str) -> float:
        """Calcula pontuação de suspeita. Valores altos → sintético."""

    def _tokenizar(self, texto: str) -> List[str]:
        """Tokenização simples por palavras, sem pontuação, em minúsculas."""
        texto = texto.lower()
        texto = texto.translate(str.maketrans("", "", string.punctuation))
        return [t for t in texto.split() if t]

    def _sentencas(self, texto: str) -> List[str]:
        """Divide o texto em sentenças."""
        return [s.strip() for s in re.split(r'[.!?]+', texto) if len(s.strip()) > 5]


# =============================================================================
# Estratégia 1 — Marcadores Léxicos de LLM  ← PRINCIPAL DISCRIMINADOR
# Ref: WO2025037142-A1; US2025307236-A1 (G06F 016/23)
#
# Detecta expressões formulaicas típicas de saída de LLM: hedges, disclaimers,
# conectivos estruturais e frases de abertura/encerramento padrão.
# =============================================================================

class MarcadoresLLMStrategy(ScoreStrategy):
    """
    Detecta marcadores léxicos típicos de saída de LLM em PT e EN.
    É a estratégia de maior poder discriminatório do ensemble.
    Cada marcador presente acumula peso proporcional à sua especificidade.
    """

    # Marcadores de alta especificidade (peso 1.0 cada)
    _MARCADORES_FORTES = [
        r"\bé importante (notar|ressaltar|destacar|mencionar|considerar)\b",
        r"\bit is important to (note|mention|consider|highlight|recognize)\b",
        r"\bé (fundamental|essencial|crucial|relevante) (compreender|entender|destacar|observar)\b",
        r"\bit is (crucial|essential|worth noting|noteworthy|imperative) that\b",
        r"\bcabe (ressaltar|destacar|mencionar|observar)\b",
        r"\bvale (ressaltar|destacar|mencionar|notar|observar)\b",
        r"\bé imprescindível\b",
        r"\bit is imperative\b",
        r"\bin conclusion\b",
        r"\bem conclusão\b",
        r"\bto summarize\b",
        r"\bin summary\b",
        r"\bpara concluir\b",
        r"\bin essence\b",
        r"\bhas (emerged|become) (an|a) (increasingly|particularly)\b",
        r"\btem se (consolidado|tornado) (uma|um)\b",
        r"\brepresenta uma (abordagem|solução|contribuição|iniciativa)\b",
        r"\boffers (several|numerous|many|various) (advantages|benefits|opportunities)\b",
    ]

    # Marcadores de média especificidade (peso 0.5 cada)
    _MARCADORES_MEDIOS = [
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\badditionally\b",
        r"\bin addition\b",
        r"\bnonetheless\b",
        r"\bnevertheless\b",
        r"\bnotwithstanding\b",
        r"\bwhereas\b",
        r"\balemais\b",
        r"\balem disso\b",
        r"\bdessa (forma|maneira|modo)\b",
        r"\bnesse contexto\b",
        r"\bin this context\b",
        r"\bit is (widely|commonly|generally) (recognized|accepted|known|understood)\b",
        r"\bé (amplamente|comumente|geralmente) (reconhecido|aceito|utilizado)\b",
        r"\bthis (approach|method|framework|technique) (enables|allows|facilitates|ensures)\b",
        r"\besta (abordagem|solução|técnica) (permite|possibilita|garante|assegura)\b",
    ]

    @property
    def nome(self) -> str:
        return "Marcadores LLM"

    def calcular(self, texto: str) -> float:
        texto_lower = texto.lower()
        pontos = 0.0

        for padrao in self._MARCADORES_FORTES:
            if re.search(padrao, texto_lower):
                pontos += 1.0

        for padrao in self._MARCADORES_MEDIOS:
            if re.search(padrao, texto_lower):
                pontos += 0.5

        # Normalizar: 2+ pontos fortes → score ≈ 1.0
        score = min(pontos / 2.5, 1.0)
        return score


# =============================================================================
# Estratégia 2 — Uniformidade de Comprimento de Sentenças
# Ref: CN119358696-A (G06F 018/21); IN202511107978-A
#
# LLMs produzem sentenças com comprimentos mais uniformes que humanos.
# Baixo coeficiente de variação (CV) → suspeito.
# =============================================================================

class UniformidadeSentencasStrategy(ScoreStrategy):
    """
    Calcula o coeficiente de variação (CV = std/mean) dos comprimentos de sentença.
    Textos humanos: CV alto (>0.45). Textos LLM: CV baixo (<0.25).
    """

    @property
    def nome(self) -> str:
        return "Uniformidade de Sentenças"

    def calcular(self, texto: str) -> float:
        sentencas = self._sentencas(texto)
        if len(sentencas) < 2:
            return 0.3  # inconclusivo mas ligeiramente suspeito

        comprimentos = [len(s.split()) for s in sentencas]
        media = np.mean(comprimentos)
        if media < 3:
            return 0.3

        cv = np.std(comprimentos) / media

        # CV < 0.15 → muito uniforme → score = 1.0
        # CV > 0.55 → variado        → score = 0.0
        score = max(0.0, min((0.55 - cv) / 0.55, 1.0))
        return round(score, 4)


# =============================================================================
# Estratégia 3 — Comprimento Médio de Palavras (Vocabulário Formal)
# Ref: WO2025037142-A1; US2024354648-A1
#
# LLMs tendem a usar vocabulário mais formal com palavras mais longas.
# Textos humanos informais usam palavras mais curtas.
# =============================================================================

class ComprimentoPalavrasStrategy(ScoreStrategy):
    """
    Calcula o comprimento médio das palavras no texto.
    LLMs: avg ≥ 6.0 chars. Humanos informais: avg ≤ 5.0 chars.
    """

    @property
    def nome(self) -> str:
        return "Vocabulário Formal"

    def calcular(self, texto: str) -> float:
        tokens = self._tokenizar(texto)
        # Ignorar palavras muito curtas (artigos, preposições)
        palavras = [t for t in tokens if len(t) > 2]
        if not palavras:
            return 0.3

        avg_len = np.mean([len(p) for p in palavras])

        # avg_len ≤ 4.5 → provavelmente humano informal → score baixo
        # avg_len ≥ 7.0 → provavelmente LLM formal      → score alto
        score = max(0.0, (avg_len - 4.5) / 3.5)
        return round(min(score, 1.0), 4)


# =============================================================================
# Estratégia 4 — Estrutura Típica de Parágrafo LLM
# Ref: WO2025037142-A1; CN119358696-A (G06F 018/21)
#
# LLMs produzem parágrafos com estrutura introdução-desenvolvimento-conclusão
# e tendem a iniciar sentenças com sujeitos formais e conectores específicos.
# =============================================================================

class EstruturaParagrafoStrategy(ScoreStrategy):
    """
    Analisa padrões estruturais de parágrafo típicos de LLM:
    - Sentenças iniciando com conectivos formais ou sujeitos impessoais
    - Presença de estrutura 3+ sentenças bem balanceadas
    - Uso de voz passiva impessoal
    """

    # Padrões de início de sentença típicos de LLM
    _INÍCIOS_LLM = [
        r"^(it is|this is|these are|the use|the development|the integration|the role)\b",
        r"^(moreover|furthermore|additionally|however|therefore|thus|consequently)\b",
        r"^(é importante|isso ocorre|este processo|esta abordagem|o uso|o desenvolvimento|a integração)\b",
        r"^(além disso|no entanto|portanto|dessa forma|nesse sentido|em virtude)\b",
        r"^(in (recent|this|the context|conclusion|summary|addition))\b",
        r"^(em (contextos|termos|suma|conclusão|resumo|virtude))\b",
    ]

    @property
    def nome(self) -> str:
        return "Estrutura de Parágrafo LLM"

    def calcular(self, texto: str) -> float:
        sentencas = self._sentencas(texto)
        if not sentencas:
            return 0.0

        n_sentencas = len(sentencas)
        # Textos muito curtos (1 sentença) → inconclusivo
        if n_sentencas < 2:
            return 0.2

        n_inícios_llm = 0
        for sent in sentencas:
            sent_lower = sent.strip().lower()
            for padrao in self._INÍCIOS_LLM:
                if re.match(padrao, sent_lower):
                    n_inícios_llm += 1
                    break

        proporcao = n_inícios_llm / n_sentencas

        # Voz passiva com "ser/estar + particípio"
        n_passiva = len(re.findall(
            r'\b(é|são|foi|foram|será|serão|está|estão)\s+\w+(?:ado|ada|ido|ida|ados|adas|idos|idas)\b',
            texto.lower()
        ))
        score_passiva = min(n_passiva / max(n_sentencas, 1) * 2, 0.4)

        score = min(proporcao * 1.2 + score_passiva, 1.0)
        return round(score, 4)


# =============================================================================
# Estratégia 5 — Diversidade de N-Gramas (Entropia de Bigramas de Palavras)
# Ref: WO2025037142-A1; US2024354648-A1
#
# Textos LLM tendem a usar combinações de palavras mais previsíveis,
# resultando em menor entropia de bigramas normalizada pela extensão do texto.
# =============================================================================

class EntropiaBigramasStrategy(ScoreStrategy):
    """
    Calcula a razão entre bigramas únicos e bigramas totais.
    Textos humanos: alta diversidade de bigramas (ratio próximo de 1.0).
    Textos LLM: bigramas mais repetitivos apesar da extensão.
    """

    @property
    def nome(self) -> str:
        return "Diversidade de Bigramas"

    def calcular(self, texto: str) -> float:
        tokens = self._tokenizar(texto)
        if len(tokens) < 4:
            return 0.3

        bigramas = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        if not bigramas:
            return 0.3

        unicos = len(set(bigramas))
        total  = len(bigramas)
        ratio  = unicos / total  # 1.0 = todos únicos (muito diverso → humano)

        # Calcular entropia normalizada
        contagens = Counter(bigramas)
        probs = [c/total for c in contagens.values()]
        entropia = -sum(p * math.log2(p) for p in probs if p > 0)
        entropia_max = math.log2(total) if total > 1 else 1.0
        entropia_norm = entropia / entropia_max if entropia_max > 0 else 0.5

        # Alto ratio + alta entropia → humano (score baixo)
        # Baixo ratio + baixa entropia → sintético (score alto)
        # Invertemos: score = 1 - entropia_norm ponderada pelo ratio
        score = (1 - entropia_norm) * 0.5 + (1 - ratio) * 0.5
        return round(min(max(score, 0.0), 1.0), 4)


# =============================================================================
# FilterEngine — Fachada (Facade Pattern)
# =============================================================================

class FilterEngine:
    """
    Motor principal do Módulo I.
    Recebe DataFrame com coluna 'texto' e produz:
      - score de contaminação por registro (0–1)
      - classificação humano/sintético
      - scores individuais por estratégia (explicabilidade)

    Padrão Facade: esconde a complexidade do ensemble ao caller (app.py).
    """

    def __init__(
        self,
        estrategias: Optional[List[ScoreStrategy]] = None,
        pesos: Optional[Dict[str, float]] = None,
        limiar: float = DEFAULT_THRESHOLD,
    ):
        if estrategias is None:
            estrategias = _estrategias_padrao()
        if pesos is None:
            pesos = CONTAMINATION_WEIGHTS

        self.estrategias = estrategias
        self.pesos = pesos
        self.limiar = limiar

        # Normalizar pesos
        total_pesos = sum(pesos.values())
        self._pesos_norm = {k: v / total_pesos for k, v in pesos.items()}
        self._chaves_pesos = list(pesos.keys())

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analisar_texto(self, texto: str) -> Dict:
        """Analisa um único texto e retorna scores detalhados."""
        scores_individuais = {}
        for estrategia, chave in zip(self.estrategias, self._chaves_pesos):
            try:
                scores_individuais[chave] = round(estrategia.calcular(texto), 4)
            except Exception:
                scores_individuais[chave] = 0.3  # fallback seguro

        score_total = sum(
            scores_individuais[k] * self._pesos_norm.get(k, 0)
            for k in self._chaves_pesos
        )
        score_total = round(min(max(score_total, 0.0), 1.0), 4)
        classificacao = "sintético" if score_total >= self.limiar else "humano"

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
        df["classificacao_coia"] = resultados.apply(lambda r: r["classificacao"])

        for chave in self._chaves_pesos:
            df[f"score_{chave}"] = resultados.apply(
                lambda r, k=chave: r["scores_detalhados"][k]
            )
        return df

    def filtrar_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retorna (corpus_limpo, corpus_sintetico)."""
        if "classificacao_coia" not in df.columns:
            df = self.analisar_dataset(df)
        limpo     = df[df["classificacao_coia"] == "humano"].copy()
        sintetico = df[df["classificacao_coia"] == "sintético"].copy()
        return limpo, sintetico

    def metricas_resumo(self, df: pd.DataFrame) -> Dict:
        """Retorna métricas resumidas após análise."""
        if "score_contaminacao" not in df.columns:
            return {}
        total       = len(df)
        n_sinteticos = (df["classificacao_coia"] == "sintético").sum()
        n_humanos    = total - n_sinteticos
        return {
            "total_registros":   total,
            "n_humanos":         int(