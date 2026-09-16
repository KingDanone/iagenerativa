"""Sessao de chat: historico curto, janela de contexto, comandos.

Formato de prompt consistente treino/inferencia:
  ### Usuário:
  <texto>

  ### Assistente:
  <texto>

Contexto != memoria persistente: /reset limpa; fechar encerra.
"""
from __future__ import annotations

PROMPT_USER = "### Usuário:\n"
PROMPT_ASST = "\n\n### Assistente:\n"

HELP = """Comandos:
  /help     mostra esta ajuda
  /history  mostra historico curto
  /reset    limpa contexto
  /stats    mostra modelo/device/temperatura/tokens
  /exit     encerra
"""


class ChatSession:
    def __init__(self, max_context_tokens: int = 512, encode_fn=None):
        self.max_context_tokens = int(max_context_tokens)
        self.encode_fn = encode_fn  # fn(text)->list[int] p/ contar tokens
        self.history: list[tuple[str, str]] = []  # (user, assistant)

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append((user, assistant))

    def reset(self) -> None:
        self.history = []

    def build_prompt(self, pending_user: str = "") -> str:
        parts: list[str] = []
        for u, a in self.history:
            parts.append(f"{PROMPT_USER}{u}{PROMPT_ASST}{a}\n")
        if pending_user:
            parts.append(f"{PROMPT_USER}{pending_user}{PROMPT_ASST}")
        return "".join(parts)

    def truncate(self) -> None:
        """Remove turnos mais antigos ate caber em max_context_tokens (se encode_fn dado)."""
        if self.encode_fn is None:
            # sem contador: limita a 6 turnos recentes
            self.history = self.history[-6:]
            return
        while self.history:
            n = len(self.encode_fn(self.build_prompt()))
            if n <= self.max_context_tokens:
                break
            self.history.pop(0)
        # se mesmo vazio+pendente estourar, o chamador trunca prompt a esquerda

    def context_tokens(self, pending_user: str = "") -> int | None:
        if self.encode_fn is None:
            return None
        return len(self.encode_fn(self.build_prompt(pending_user)))

    def render_history(self, limit: int = 6) -> str:
        lines = []
        for u, a in self.history[-limit:]:
            lines.append(f"Você: {u}\nIA: {a}\n")
        return "\n".join(lines) if lines else "(historico vazio)"
