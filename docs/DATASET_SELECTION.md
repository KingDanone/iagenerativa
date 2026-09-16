# Seleção de Datasets — mini_ia_generativa

Pesquisa realizada em 2026-09-13. Critérios: qualidade textual, volume de português,
diversidade, documentação, origem identificável, licença/termos, streaming, limite de tamanho.

## Candidatos avaliados

| Nome | Fonte | Idioma | Tamanho | Formato | Licença/Termos | Streaming? | Limite? | Vantagens | Desvantagens |
|---|---|---|---|---|---|---|---|---|---|
| Wikipedia PT | `wikimedia/wikipedia` (HF) / dumps.wikimedia.org | PT (majorit. BR+PT) | ~1–2 GB texto limpo | parquet/json via `datasets` | CC BY-SA 4.0 (atribuição + share-alike) | sim (`streaming=True`) | sim (n docs / bytes) | alta qualidade, educacional, documentado, estável | pouco diálogo; viés enciclopédico |
| Corpus Carolina (USP/C4AI) | `carolina-c4ai/corpus-carolina` (HF) + sites.usp.br/corpuscarolina | PT-BR contemporâneo | ~11 GB / 823M tokens (Ada 1.2) | XML/TEI + HF parquet | múltiplas licenças por documento (ver metadados; questão a verificar por doc) | sim (HF) | sim | PT-BR geral, proveniência/tipologia, robusto | licenças heterogêneas; 11 GB exige amostragem |
| OSCAR-2201 pt | `oscar-corpus/OSCAR-2201`, config `pt` | PT (web) | ~170 GB brutos, 23M docs | jsonl/parquet via `datasets` | CC0-1.0 (dados) — atenção: pode conter dados pessoais/sensíveis do Common Crawl | sim | sim | volume enorme, diversidade web | ruidoso (spam/HTML/duplicatas); requer filtragem forte |
| CulturaX / mC4-pt | HF `uonlp/CulturaX`, `allenai/c4` (multilingual) | PT entre 100+ línguas | TBs (usar fração) | parquet | ODC-BY (mC4) / misto (CulturaX) | sim | sim | já filtrado, multilíngue | PT é fração; qualidade variável |
| BrWAC | corpus web PT (usado no PTT5) | PT-BR | ~2–3 GB texto | texto | acadêmico (verificar termos) | parcial | sim | PT-BR web de qualidade | menos mantido que Carolina |
| QA-Portuguese (500k pares) | `Jpzinn654/qa-portuguese-small` (derivado MQA) | PT | ~500k pares pergunta/resposta | HF dataset | MIT (repo) — origem MQA a verificar | sim | sim | formato conversacional/QA pronto | qualidade variável; possível ruído de tradução |
| Wikipedia-PT-BR-Instruct-5k | `br-llm-data/wikipedia-pt-br-instruct-5k` | PT-BR | ~53k exemplos | HF (messages/prompt_completion/alpaca) | "other" (sintético; verificar) | sim | sim | instrução em PT-BR | **sintético** (gerador Gemma) — usar no máx. como fração pequena |
| Seed próprio | `data/seed_conversational.txt` (este repo) | PT-BR | ~12 exemplos autorais | txt | autoria própria (livre) | n/a | n/a | formato `### Usuário:/### Assistente:` exato, limpo | pequeno — apenas sinal de formato, não conhecimento |

Descartados para LM geral: datasets de sentimento (B2W/Amazon/Steam, curtos e opinativos),
áudio/TTS (não-texto), preferências RLHF pagas (licença comercial), MedPT (domínio médico
restrito — útil só como expansão futura opcional).

## Decisão

Composição-alvo (~ref. 40–60 / 15–25 / 10–20 / 10–20):

- **~55% português geral**: Wikipedia PT (base confiável) + amostra Carolina/OSCAR filtrada.
- **~20% educacional/científico**: Wikipedia (ciência/história) — subconjunto natural da fonte.
- **~10% literatura/documentos**: Carolina (domínio público/literário) + seed.
- **~15% conversacional**: QA-Portuguese (fração) + seed próprio no formato exato do chat.

Pipeline (`scripts/download_data.py`): streaming por fonte, orçamento em bytes por fonte
(`--max-gb`), `clean_data.py` (NFC, HTML, lixo, dedup sha256 normalizado, filtro PT),
`prepare_data.py` (manifesto), `train_tokenizer.py` (BPE 4096), `build_dataset.py` (uint16).

## Estratégia progressiva

1. Fase 1: ~5–20 MB (validar pipeline + overfit).
2. Fase 2: ~100 MB (teste real).
3. Fase 3: ~2 GB (treinamento principal); 5–10 GB só se houver ganho medido.

## Licenças — questões a verificar

- Carolina: respeitar licença por documento (metadados TEI); marcar como questão a verificar
  antes de redistribuir textos (não redistribuímos corpus — só manifesto + scripts).
- OSCAR: CC0 nos dados, mas Common Crawl pode conter dados pessoais — filtragem + não memorizar.
- Não incluímos nenhum corpus gigante no Git (só manifesto + seed).
