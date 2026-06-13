# =============================================================================
# CO-IA — Módulo III: Auditoria de Proveniência
# modulo3_provenance.py
#
# Registra a origem verificável de cada amostra via hash SHA-256, protege um
# conjunto "padrão-ouro" (dados humanos verificados) e recomenda o reequilíbrio
# do corpus para mitigar Data Poisoning.
#
# Fundamentação em patentes:
#   • US2025238634-A1  — proteção de origem de dados de treinamento
#   • US2026080037-A1  — curadoria com proveniência verificável
#   • BR 11 2026 0105  — proteção de direitos autorais em IA (G06F 21/16)
#   • BR 11 2023 0065  — rastreamento de dados compartilhados (G06N 5/02)
#   • US2025307465-A1  — segurança e integridade de dados (G06F 021/60)
#
# Padrões de projeto:
#   • Repository     — ProvenanceRepository gerencia o ciclo de vida dos registros
#   • Value Object   — ProvenanceRecord é imutável após criação
#   • Builder        — ReportBuilder constrói o relatório de forma incremental
# =============================================================================

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

from config import (
    HASH_ALGORITHM,
    GOLD_STANDARD_MIN_RATIO,
    SYNTHETIC_MAX_RATIO,
    COLOR_HUMANO,
    COLOR_SINTETICO,
    COLOR_ALERTA,
)


# =============================================================================
# Value Object: ProvenanceRecord
# Imutável após criação (dataclass frozen). Garante rastreabilidade completa.
# Ref: US2025238634-A1 — estrutura de registro de origem
# =============================================================================

@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Registro de proveniência de um único texto no corpus.
    Campos seguem a estrutura proposta em US2025238634-A1 e BR 11 2026 0105.
    """
    record_id:   str          # Identificador único (coluna 'id' do CSV)
    hash_sha256: str          # SHA-256 do conteúdo textual
    fonte:       str          # Origem declarada (arquivo, URL, autor)
    data_ingestao: str        # Timestamp de ingestão no pipeline
    tipo_declarado: str       # 'humano' ou 'sintético' (pelo CO-IA)
    score_contaminacao: float # Score do Módulo I
    is_gold_standard: bool    # Se pertence ao conjunto padrão-ouro protegido
    comprimento_chars: int    # Comprimento do texto em caracteres
    comprimento_tokens: int   # Comprimento em tokens (palavras)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Repository: ProvenanceRepository
# Armazena, consulta e gerencia registros de proveniência.
# Ref: BR 11 2023 0065 (G06N 5/02) — rastreamento de dados compartilhados
# =============================================================================

class ProvenanceRepository:
    """
    Gerencia o ciclo de vida dos registros de proveniência.
    Padrão Repository: isola a lógica de armazenamento do resto do sistema.
    """

    def __init__(self):
        self._registros: Dict[str, ProvenanceRecord] = {}
        self._gold_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # Operações de escrita
    # ------------------------------------------------------------------

    def adicionar(self, record: ProvenanceRecord) -> None:
        """Adiciona ou substitui um registro."""
        self._registros[record.record_id] = record
        if record.is_gold_standard:
            self._gold_ids.add(record.record_id)

    def marcar_gold_standard(self, ids: List[str]) -> int:
        """
        Marca um conjunto de IDs como padrão-ouro protegido.
        Ref: US2025238634-A1 — proteção de conjunto gold standard.
        Retorna número de registros marcados com sucesso.
        """
        marcados = 0
        for rid in ids:
            if rid in self._registros:
                # Recriar com is_gold_standard=True (frozen → nova instância)
                r = self._registros[rid]
                novo = ProvenanceRecord(
                    record_id=r.record_id,
                    hash_sha256=r.hash_sha256,
                    fonte=r.fonte,
                    data_ingestao=r.data_ingestao,
                    tipo_declarado=r.tipo_declarado,
                    score_contaminacao=r.score_contaminacao,
                    is_gold_standard=True,
                    comprimento_chars=r.comprimento_chars,
                    comprimento_tokens=r.comprimento_tokens,
                )
                self._registros[rid] = novo
                self._gold_ids.add(rid)
                marcados += 1
        return marcados

    # ------------------------------------------------------------------
    # Operações de leitura
    # ------------------------------------------------------------------

    def todos(self) -> List[ProvenanceRecord]:
        return list(self._registros.values())

    def gold_standard(self) -> List[ProvenanceRecord]:
        return [r for r in self._registros.values() if r.is_gold_standard]

    def sinteticos(self) -> List[ProvenanceRecord]:
        return [r for r in self._registros.values() if r.tipo_declarado in ("sintetico", "sintético")]

    def humanos(self) -> List[ProvenanceRecord]:
        return [r for r in self._registros.values() if r.tipo_declarado == "humano"]

    def buscar_por_hash(self, hash_val: str) -> Optional[ProvenanceRecord]:
        for r in self._registros.values():
            if r.hash_sha256 == hash_val:
                return r
        return None

    def to_dataframe(self) -> pd.DataFrame:
        if not self._registros:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._registros.values()])


# =============================================================================
# ProvenanceEngine — orquestrador do Módulo III
# Ref: US2026080037-A1 — pipeline de curadoria com proveniência verificável
# =============================================================================

class ProvenanceEngine:
    """
    Motor principal do Módulo III.
    Gera registros de proveniência para cada texto do corpus,
    gerencia o padrão-ouro e recomenda reequilíbrio.
    """

    def __init__(self):
        self.repository = ProvenanceRepository()

    # ------------------------------------------------------------------
    # Geração de hash — núcleo da proveniência
    # Ref: BR 11 2026 0105 (G06F 21/16) — identificação criptográfica de conteúdo
    # ------------------------------------------------------------------

    @staticmethod
    def gerar_hash(texto: str) -> str:
        """Gera SHA-256 do texto. Imutável e verificável."""
        return hashlib.new(HASH_ALGORITHM, texto.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Ingestão de DataFrame do Módulo I
    # ------------------------------------------------------------------

    def ingerir_dataset(self, df: pd.DataFrame) -> ProvenanceRepository:
        """
        Recebe o DataFrame já analisado pelo Módulo I e gera registros
        de proveniência para cada registro.
        Retorna o repositório preenchido.
        """
        timestamp = datetime.now().isoformat()

        for _, row in df.iterrows():
            texto = str(row.get("texto", ""))
            tokens = texto.split()

            record = ProvenanceRecord(
                record_id=str(row.get("id", f"REC_{_}")),
                hash_sha256=self.gerar_hash(texto),
                fonte=str(row.get("fonte", "desconhecida")),
                data_ingestao=timestamp,
                tipo_declarado=str(row.get("classificacao_coia", row.get("tipo_real", "desconhecido"))),
                score_contaminacao=float(row.get("score_contaminacao", 0.0)),
                is_gold_standard=False,
                comprimento_chars=len(texto),
                comprimento_tokens=len(tokens),
            )
            self.repository.adicionar(record)

        return self.repository

    # ------------------------------------------------------------------
    # Recomendação de reequilíbrio
    # Ref: WO2025037142-A1 (NEC Lab) — proporção saudável de dados
    #      US2025307236-A1             — rebalanceamento automático
    # ------------------------------------------------------------------

    def recomendar_reequilibrio(self) -> Dict:
        """
        Analisa a composição atual do corpus e recomenda ações de reequilíbrio
        para manter a proporção saudável humano:sintético.
        """
        total = len(self.repository.todos())
        if total == 0:
            return {"status": "corpus vazio"}

        n_humanos   = len(self.repository.humanos())
        n_sinteticos = len(self.repository.sinteticos())
        n_gold       = len(self.repository.gold_standard())

        proporcao_sint = n_sinteticos / total
        proporcao_hum  = n_humanos / total
        proporcao_gold = n_gold / total

        # Status geral
        if proporcao_sint > SYNTHETIC_MAX_RATIO:
            status = "⚠️ Desequilíbrio — excesso de dados sintéticos"
            acao = f"Remover ou neutralizar pelo menos {int((proporcao_sint - SYNTHETIC_MAX_RATIO) * total)} registros sintéticos."
        elif proporcao_gold < GOLD_STANDARD_MIN_RATIO:
            status = "🔶 Atenção — padrão-ouro insuficiente"
            min_total = int(total * GOLD_STANDARD_MIN_RATIO)
            n_needed  = max(0, min_total - n_gold)
            acao = f"Voce marcou {n_gold} registros ({round(proporcao_gold*100,1)}%). O minimo recomendado e {min_total} registros (30%). Marque mais {n_needed} registros humanos como padrao-ouro."
        else:
            status = "✅ Corpus equilibrado"
            acao = "Nenhuma ação imediata necessária. Continue monitorando."

        return {
            "total_registros":      total,
            "n_humanos":            n_humanos,
            "n_sinteticos":         n_sinteticos,
            "n_gold_standard":      n_gold,
            "proporcao_sinteticos": round(proporcao_sint * 100, 1),
            "proporcao_humanos":    round(proporcao_hum * 100, 1),
            "proporcao_gold":       round(proporcao_gold * 100, 1),
            "status":               status,
            "acao_recomendada":     acao,
            "limite_sinteticos":    round(SYNTHETIC_MAX_RATIO * 100, 0),
            "minimo_gold":          round(GOLD_STANDARD_MIN_RATIO * 100, 0),
        }


# =============================================================================
# ReportBuilder — constrói o relatório de proveniência em Markdown
# Ref: US2026080037-A1 — exportação de relatório auditável
# =============================================================================

class ReportBuilder:
    """
    Builder incremental para o relatório de proveniência.
    Produz um documento Markdown completo, pronto para apresentação
    no pitch acadêmico e para depósito no INPI/DIT.
    """

    def __init__(self):
        self._secoes: List[str] = []

    def adicionar_cabecalho(self, titulo: str, subtitulo: str, versao: str) -> "ReportBuilder":
        self._secoes.append(
            f"# {titulo}\n\n"
            f"> {subtitulo}\n\n"
            f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n---\n"
        )
        return self

    def adicionar_resumo_execucao(self, metricas_m1: Dict, risco_m2: Dict, reequilibrio: Dict) -> "ReportBuilder":
        self._secoes.append(
            f"## 1. Resumo da Execução\n\n"
            f"| Indicador | Valor |\n"
            f"|-----------|-------|\n"
            f"| Total de registros analisados | **{metricas_m1.get('total_registros', 0)}** |\n"
            f"| Registros classificados como humanos | **{metricas_m1.get('n_humanos', 0)}** |\n"
            f"| Registros classificados como sintéticos | **{metricas_m1.get('n_sinteticos', 0)}** |\n"
            f"| Taxa de contaminação | **{metricas_m1.get('taxa_contaminacao', 0.0)}%** |\n"
            f"| Score médio de contaminação | **{metricas_m1.get('score_medio', 0.0)}** |\n"
            f"| Risco de colapso (Módulo II) | **{risco_m2.get('zona', 'N/A')}** ({risco_m2.get('risco_total', 0.0):.2f}) |\n"
            f"| Status do corpus | **{reequilibrio.get('status', 'N/A')}** |\n\n"
        )
        return self

    def adicionar_registros_proveniencia(self, df_prov: pd.DataFrame) -> "ReportBuilder":
        if df_prov.empty:
            self._secoes.append("## 2. Registros de Proveniência\n\n_Nenhum registro disponível._\n\n")
            return self

        # Tabela simplificada
        cols = ["record_id", "hash_sha256", "fonte", "tipo_declarado", "score_contaminacao", "is_gold_standard"]
        cols_existentes = [c for c in cols if c in df_prov.columns]
        tabela = df_prov[cols_existentes].copy()
        if "hash_sha256" in tabela.columns:
            tabela["hash_sha256"] = tabela["hash_sha256"].str[:16] + "..."

        md_tabela = tabela.to_markdown(index=False)
        self._secoes.append(f"## 2. Registros de Proveniência\n\n{md_tabela}\n\n")
        return self

    def adicionar_reequilibrio(self, reequilibrio: Dict) -> "ReportBuilder":
        self._secoes.append(
            f"## 3. Análise de Reequilíbrio do Corpus\n\n"
            f"| Composição | Valor |\n"
            f"|------------|-------|\n"
            f"| Dados humanos | **{reequilibrio.get('proporcao_humanos', 0)}%** ({reequilibrio.get('n_humanos', 0)} registros) |\n"
            f"| Dados sintéticos | **{reequilibrio.get('proporcao_sinteticos', 0)}%** ({reequilibrio.get('n_sinteticos', 0)} registros) |\n"
            f"| Padrão-ouro marcado | **{reequilibrio.get('proporcao_gold', 0)}%** ({reequilibrio.get('n_gold_standard', 0)} registros) |\n"
            f"| Limite recomendado (sintéticos) | ≤ **{reequilibrio.get('limite_sinteticos', 50)}%** |\n"
            f"| Mínimo padrão-ouro | ≥ **{reequilibrio.get('minimo_gold', 30)}%** |\n\n"
            f"**Status:** {reequilibrio.get('status', '')}\n\n"
            f"**Ação recomendada:** {reequilibrio.get('acao_recomendada', '')}\n\n"
        )
        return self

    def adicionar_fundamentacao(self) -> "ReportBuilder":
        self._secoes.append(
            "## 4. Fundamentação Técnica — Patentes de Referência\n\n"
            "| Patente | Titular | Técnica Implementada |\n"
            "|---------|---------|----------------------|\n"
            "| WO2025037142-A1  | NEC Lab             | Curadoria e qualidade de dados sintéticos |\n"
            "| IN202511107978-A | Univ. Manipal       | Detecção ML de dados sintéticos |\n"
            "| US2025342187-A1  | Madisetti V.        | Sistema multi-nível com feedback loop |\n"
            "| US2024411789-A1  | Madisetti V.        | Refinamento por geração e baseline |\n"
            "| US2025238634-A1  | —                   | Proteção de origem de dados |\n"
            "| US2026080037-A1  | —                   | Curadoria com proveniência verificável |\n"
            "| BR 11 2026 0105  | —                   | Proteção de direitos autorais IA (G06F 21/16) |\n"
            "| BR 11 2023 0065  | —                   | Rastreamento de dados (G06N 5/02) |\n\n"
            "> Mapeamento sistemático: 244 estudos científicos (MSL estado da arte) + "
            "71 patentes (MST estado da técnica) — UNIRIO / BSI, 2025.\n\n"
        )
        return self

    def adicionar_rodape(self) -> "ReportBuilder":
        # Rodapé minimalista — sem texto repetitivo
        self._secoes.append("---\n")
        return self

    def build(self) -> str:
        """Retorna o relatório completo como string Markdown."""
        return "\n".join(self._secoes)


# =============================================================================
# Ponto de entrada público do Módulo III
# =============================================================================

def gerar_proveniencia(
    df: pd.DataFrame,
    ids_gold: Optional[List[str]] = None,
) -> Dict:
    """
    Ponto de entrada do Módulo III.
    Recebe DataFrame com resultados do Módulo I e retorna:
      - repositório de proveniência
      - DataFrame de proveniência
      - recomendação de reequilíbrio
    """
    engine = ProvenanceEngine()
    repo = engine.ingerir_dataset(df)

    if ids_gold:
        engine.repository.marcar_gold_standard(ids_gold)

    reequilibrio = engine.recomendar_reequilibrio()
    df_prov = repo.to_dataframe()

    return {
        "engine":       engine,
        "repository":   repo,
        "df_provenance": df_prov,
        "reequilibrio": reequilibrio,
    }
