# Chat

## Formato de prompt (idêntico ao treino)

```
### Usuário:
<mensagem>

### Assistente:
<resposta>
```

Os dados conversacionais foram serializados exatamente assim antes da
tokenização, então o modelo aprende o formato como continuação natural.

## Contexto curto (≠ memória persistente)

- `ChatSession` guarda últimos turnos e monta o prompt concatenado.
- Limite `max_context_tokens` (padrão 512): estouro remove turnos **mais antigos**.
- `/reset` limpa; fechar o programa apaga tudo. Nada é gravado em disco.
  (Requisito de privacidade da especificação.)

## Geração

Amostragem com `temperature` (padrão 0.8) + `top-k` (padrão 40);
para ao gerar `<EOS>` ou atingir `max_new_tokens`. Sem respostas
hardcoded: todo texto exibido como "IA" sai do `model.generate`
(ver `src/generation.py`). Greedy (`temperature=0`) só para depuração —
tende a repetir.

## Comandos

`/help` `/history` `/reset` `/stats` `/exit` (implementados em `chat.py`).

## Limitações honestas

Modelo de 17M params / contexto 256: mantém **poucos turnos** de coerência
(ex.: nome dito 1–2 turnos atrás). Teste `/stats` mostra tokens de contexto
reais. Falhas de memória além da janela são esperadas e documentadas
em `docs/RESULTS.md` — não são "bugs do chat", são limites do modelo.
