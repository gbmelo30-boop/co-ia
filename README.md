# CO-IA — Curadoria de Origem para IA

> Pipeline integrado de detecção, monitoramento e proveniência para mitigação do Model Collapse em sistemas de IA Generativa.

**Autor:** Gabriel de Melo Guedes Souza  
**Instituição:** UNIRIO — Bacharelado em Sistemas de Informação (BSI)  
**Disciplina:** Metodologia Científica e Tecnológica (MCT)  
**Versão:** 1.0.0-MVP

---

## O Problema

O **Model Collapse** ocorre quando modelos de IA Generativa são treinados recursivamente sobre dados sintéticos gerados por versões anteriores do próprio modelo. O resultado é uma degradação progressiva em acurácia e diversidade — o chamado efeito *"Habsburg AI"* — com perda da "cauda longa" da distribuição dos dados originais.

## A Solução — CO-IA

O CO-IA é um framework integrado com três módulos complementares, cada um fundamentado em patentes mapeadas sistematicamente:

| Módulo | Função | Patentes de Referência |
|--------|--------|------------------------|
| **I — Filtro de Entrada** | Calcula *contamination score* (0–1) por registro e classifica como humano/sintético | WO2025037142-A1, IN202511107978-A, CN119358696-A, US2024354648-A1 |
| **II — Monitoramento** | Métricas de diversidade do corpus + simulação de N gerações recursivas | US2025342187-A1, US2025094459-A1, US2025217394-A1, US2024411789-A1 |
| **III — Proveniência** | Hash SHA-256 por registro + padrão-ouro + reequilíbrio | US2025238634-A1, US2026080037-A1, BR 11 2026 0105, BR 11 2023 0065 |

> Fundamentado em **71 patentes** (Derwent Innovation + INPI) e **244 estudos científicos** mapeados sistematicamente.

---

## Instalação Local

```bash
# Clonar o repositório
git clone https://github.com/gbmelo30-boop/co-ia.git
cd co-ia

# Instalar dependências
pip install -r requirements.txt

# Rodar o dashboard
streamlit run app.py
```

Acesse em: `http://localhost:8501`

---

## Estrutura do Projeto

```
co-ia/
├── app.py                          # Dashboard principal (Streamlit)
├── config.py                       # Configurações globais e constantes
├── requirements.txt
├── README.md
├── modules/
│   ├── modulo1_filtro.py           # Módulo I — Filtro de Entrada
│   ├── modulo2_monitor.py          # Módulo II — Monitoramento de Degeneração
│   └── modulo3_provenance.py       # Módulo III — Auditoria de Proveniência
└── data/
    └── demo_dataset.csv            # Dataset de demonstração (24 registros rotulados)
```

### Padrões de Projeto Utilizados

- **Strategy** — cada estratégia de pontuação do Módulo I é intercambiável (`ScoreStrategy`)
- **Facade** — `FilterEngine`, `analisar_corpus()` e `gerar_proveniencia()` expõem APIs simples
- **Repository** — `ProvenanceRepository` gerencia o ciclo de vida dos registros
- **Builder** — `ReportBuilder` constrói o relatório de forma incremental
- **Factory** — `build_default_engine()` instancia o pipeline padrão
- **Value Object** — `ProvenanceRecord` é imutável após criação (`frozen=True`)

---

## Deploy — Streamlit Community Cloud

1. Faça fork ou push deste repositório para sua conta GitHub
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud)
3. Clique em **New app** → selecione o repositório → `app.py` → **Deploy**
4. Aguarde ~2 minutos → URL pública gerada automaticamente

---

## Formato do CSV de Upload

```csv
id,texto,fonte,tipo_real
H001,"Texto do registro 1","Blog pessoal","humano"
S001,"Texto do registro 2","Gerado por LLM","sintetico"
```

Somente a coluna `texto` é obrigatória. As demais são opcionais.

---

## Licença

Software acadêmico — UNIRIO / BSI, 2025. Para registro no INPI/DIT, consulte o resumo técnico exportável pelo próprio dashboard.
