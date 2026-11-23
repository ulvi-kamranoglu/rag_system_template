"""Exercise the real preparation/embedding boundary without model downloads."""
import hashlib

import polars as pl
import pytest

from chunking_pipeline.chunking import build_chunks
from chunking_pipeline.preparation import prepare
from embedding.config import CHUNKS_INPUT_PATH, DOCUMENT_MAX_LENGTH
from embedding.pipeline import build_mapping, validate_input


class ByteTokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = list(text.encode("utf-8"))
        return [1, *tokens, 2] if add_special_tokens else tokens

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("utf-8")

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_chunks_feed_embedding_and_preserve_document_identity(tmp_path):
    source = pl.DataFrame([
        {"id": "page:15", "source_type": "page", "title": "Admissions",
         "content": "Applications are open. " * 35, "change_type": "UPDATED"},
        {"id": "15", "source_type": "post", "title": "News",
         "content": "Welcome to the university.", "change_type": "NEW"},
    ])
    chunks = build_chunks(prepare(source), ByteTokenizer(), DOCUMENT_MAX_LENGTH, 64)
    path = tmp_path / CHUNKS_INPUT_PATH.name
    chunks.write_parquet(path)
    loaded = pl.read_parquet(path)
    validate_input(loaded)
    mapping = build_mapping(loaded)
    assert set(mapping["document_id"]) == {"page:15", "post:15"}
    assert mapping["chunk_id"].n_unique() == mapping.height
    assert mapping["embedding_tokens"].max() <= DOCUMENT_MAX_LENGTH
    assert mapping["vector_index"].to_list() == list(range(mapping.height))
    assert set(mapping["change_type"]) == {"NEW", "UPDATED"}
    assert mapping.group_by("document_id").agg(pl.col("retrieval_hash").n_unique())["retrieval_hash"].max() == 1


def test_title_changes_retrieval_hash_without_changing_identity():
    def chunk(title):
        return build_chunks(prepare(pl.DataFrame([{
            "id": "page:15", "source_type": "page", "title": title,
            "content": "Applications are open.",
        }])), ByteTokenizer(), DOCUMENT_MAX_LENGTH, 64).row(0, named=True)
    before, after = chunk("Old title"), chunk("New title")
    assert before["document_id"] == after["document_id"]
    assert before["content_hash"] == after["content_hash"]
    assert before["retrieval_hash"] != after["retrieval_hash"]
    assert before["content_hash"] == hashlib.sha256(b"Applications are open.").hexdigest()


def test_published_chunks_satisfy_embedding_contract():
    if not CHUNKS_INPUT_PATH.exists():
        pytest.skip("No published dataset in this checkout")
    chunks = pl.read_parquet(CHUNKS_INPUT_PATH)
    validate_input(chunks)
    assert chunks["embedding_tokens"].max() <= DOCUMENT_MAX_LENGTH
    assert chunks["document_id"].null_count() == 0
