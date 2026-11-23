import polars as pl
from chunking_pipeline.preparation import clean_retrieval_text, prepare
from chunking_pipeline.chunking import sliding_windows, build_chunks


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        ids = list(text.encode("utf-8"))
        return [1] + ids + [2] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return bytes(ids).decode("utf-8")

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_cleaning():
    assert clean_retrieval_text("<p>Hello&nbsp;world</p>") == "Hello world"


def test_late_exclusion_reason_and_optional_metadata():
    rows = [{"id": str(i), "source_type": "form", "title": "Question",
             "content": "Useful university information.", "metadata": None}
            for i in range(120)]
    rows.append({"id": "120", "source_type": "form", "title": "Question",
                 "content": "Additional useful information.", "metadata": "{}"})
    rows.append({"id": "121", "source_type": "page", "title": "example subdomain",
                 "content": "Site information.", "metadata": None})
    source = pl.DataFrame(rows, infer_schema_length=None)
    prepared = prepare(source)
    assert prepared.schema["exclusion_reason"] == pl.String
    assert prepared["exclusion_reason"][-1] == "low_value_site_page"
    assert prepared["exclusion_reason"].null_count() == 121
    chunks = build_chunks(prepared, FakeTokenizer(), 384, 64)
    assert chunks.height == 121
    assert chunks.schema["metadata"] == pl.String
    assert chunks["metadata"][-1] == "{}"


def test_utc_parquet_dates_survive_preparation_and_chunking(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    timestamp = datetime(2026, 10, 22, 8, 0, tzinfo=ZoneInfo("UTC"))
    source = pl.DataFrame([{
        "id": "15", "source_type": "form", "title": "Where?",
        "content": "In the library.", "date": timestamp, "modified": timestamp,
    }])
    path = tmp_path / "canonical_dataset.parquet"
    source.write_parquet(path)
    prepared = prepare(pl.read_parquet(path))
    chunks = build_chunks(prepared, FakeTokenizer(), 128, 16)
    assert chunks["date"][0] == timestamp
    assert chunks["modified"][0] == timestamp


def test_windows():
    assert sliding_windows(list(range(12)), 6, 2) == [list(range(6)), list(range(4, 10)), list(range(8, 12))]


def test_faq_prefix_and_source_id():
    df = pl.DataFrame([{"id": "15", "source_type": "form", "title": "Where?", "content": "In the library.", "search_text": "Where?"}])
    prepared = prepare(df)
    output = build_chunks(prepared, FakeTokenizer(), 128, 16)
    assert output["chunk_id"][0] == "form:15:chunk:0000"
    assert output["chunk_text"][0] == "In the library."
    assert output["embedding_text"][0].startswith("Title: Where?\n\nContent:\n")
    assert output["embedding_tokens"].max() <= 128


def test_preserves_sentence_and_paragraph_boundaries():
    from chunking_pipeline.chunking import boundary_aware_chunks, token_count
    tokenizer = FakeTokenizer()
    text = "First sentence. Second sentence.\n\nThird paragraph starts here. Fourth sentence."
    chunks = boundary_aware_chunks(text, tokenizer, "", max_tokens=48, overlap_tokens=0)
    assert chunks == ["First sentence. Second sentence.", "Third paragraph starts here. Fourth sentence."]
    assert all(token_count(tokenizer, c) <= 48 for c in chunks)


def test_oversized_sentence_fallback_keeps_words():
    from chunking_pipeline.chunking import boundary_aware_chunks, token_count
    tokenizer = FakeTokenizer()
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = boundary_aware_chunks(text, tokenizer, "", max_tokens=20, overlap_tokens=0)
    assert " ".join(chunks) == text
    assert all(token_count(tokenizer, c) <= 20 for c in chunks)


def test_overlap_reuses_complete_sentences():
    from chunking_pipeline.chunking import boundary_aware_chunks, token_count
    tokenizer = FakeTokenizer()
    text = "One two. Three four. Five six. Seven eight."
    chunks = boundary_aware_chunks(text, tokenizer, "", max_tokens=26, overlap_tokens=12)
    assert chunks == ["One two. Three four.", "Three four. Five six.", "Five six. Seven eight."]


def test_long_title_reduces_overlap_to_available_budget():
    title = 'T' * 310
    content = ' '.join(f'word{i}.' for i in range(50))
    prepared = prepare(pl.DataFrame([{'id': 'news:1', 'source_type': 'news', 'title': title, 'content': content}]))
    result = build_chunks(prepared, FakeTokenizer(), 384, 64)
    assert result.height > 1
    assert result['embedding_tokens'].max() <= 384
    assert result['overlap_tokens_config'].unique().to_list() == [64]
    assert result['overlap_tokens_effective'].max() < 49
    assert all(f'word{i}.' in ' '.join(result['chunk_text']) for i in range(50))
    assert result['title'].unique().to_list() == [title]


def test_oversized_title_is_chunked_without_losing_title_or_body():
    import re
    title = 'Long title sentence. ' * 100
    content = 'The body must also remain completely available.'
    prepared = prepare(pl.DataFrame([{'id': 'news:2', 'source_type': 'news', 'title': title, 'content': content}]))
    result = build_chunks(prepared, FakeTokenizer(), 384, 0)
    assert result['embedding_tokens'].max() <= 384
    actual = re.sub(r'\s+', ' ', ' '.join(result['chunk_text'])).strip()
    assert actual == re.sub(r'\s+', ' ', title + '\n\n' + content).strip()
    assert result['title'].unique().to_list() == [title]
