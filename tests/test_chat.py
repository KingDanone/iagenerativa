from src.chat import ChatSession


def test_prompt_format_and_truncate():
    enc = lambda t: t.split()  # contador fake por palavras
    s = ChatSession(max_context_tokens=10, encode_fn=enc)
    s.add_turn("Olá.", "Olá! Como posso ajudar?")
    s.add_turn("Meu nome é João.", "Prazer, João.")
    p = s.build_prompt("Qual é o meu nome?")
    assert "### Usuário:" in p and "### Assistente:" in p
    assert p.rstrip().endswith("### Assistente:")
    # forca truncamento
    for i in range(10):
        s.add_turn(f"pergunta longa numero {i} com varias palavras extras", "resposta")
    s.truncate()
    assert s.context_tokens() <= 10


def test_reset_clears():
    s = ChatSession()
    s.add_turn("a", "b")
    s.reset()
    assert s.history == [] and "### Usuário:" not in s.build_prompt()
