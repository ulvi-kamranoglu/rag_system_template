import json
from assistant_app.llm.repetition import RepetitionGuard
from assistant_app.llm import client


def collect(chunks):
    guard = RepetitionGuard()
    emitted = []
    for text in chunks:
        emitted.append(guard.feed(text))
    emitted.append(guard.finish())
    return "".join(emitted), guard.stopped


def test_literal_loop_split_across_tokens_stops_without_leaking_loop():
    text = "Professor Rövnəq Rzayev prorektordur. prof." + "dur." * 100
    output, stopped = collect(list(text))
    assert stopped
    assert output.startswith("Professor Rövnəq Rzayev prorektordur.")
    assert output.count("dur.") <= 2


def test_normal_long_answer_is_preserved_exactly():
    text = "\n".join(f"{n}. UNEC kitabxanasında {n} saylı xidmət haqqında məlumat." for n in range(30))
    output, stopped = collect([text[i:i+17] for i in range(0, len(text), 17)])
    assert output == text
    assert not stopped


def test_whitespace_and_punctuation_are_not_classified_as_word_loops():
    text = "Başlıq\n" + "-" * 100 + "\n" + " " * 100 + "Son."
    assert collect([text]) == (text, False)


def test_chat_closes_upstream_and_streamed_text_matches_returned_text(monkeypatch):
    consumed = []
    class Response:
        closed = False
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
        def raise_for_status(self): pass
        def iter_lines(self):
            for text in ["Cavab: prof.", "dur." * 8, "should never be consumed"]:
                consumed.append(text)
                yield json.dumps({"message": {"content": text}}).encode()
    response = Response()
    monkeypatch.setattr(client.requests, "post", lambda *a, **kw: response)
    tokens = []
    answer = client.chat([], .2, tokens.append)
    assert response.closed
    assert len(consumed) == 2
    assert answer == "".join(tokens).strip()
    assert "should never" not in answer


def test_json_router_and_not_found_remain_intact(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_lines(self):
            yield json.dumps({"message":{"content":"NOT_FOUND"}}).encode()
    monkeypatch.setattr(client.requests,"post",lambda *a,**kw:Response())
    assert client.chat([],0) == "NOT_FOUND"
    assert client.chat([],0,json_output=True) == "NOT_FOUND"
