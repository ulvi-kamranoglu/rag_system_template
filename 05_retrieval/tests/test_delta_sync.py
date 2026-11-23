import polars as pl

from retrieval.delta_sync_pipeline import (
    validate_delta_mapping,
    build_updated_corpus,
)


def test_delta_mapping_validation():

    df = pl.DataFrame(
        {
            "document_id": [
                "doc_002",
                "doc_003",
            ],
            "chunk_id": [
                "doc_002_chunk_0000",
                "doc_003_chunk_0000",
            ],
            "change_type": [
                "UPDATED",
                "NEW",
            ],
            "retrieval_hash": [
                "hash_doc_002_new",
                "hash_doc_003",
            ],
        }
    )

    # Error verməməlidir
    validate_delta_mapping(df)


def test_updated_document_replaces_old_chunks():

    existing = pl.DataFrame(
        {
            "document_id": [
                "doc_001",
                "doc_002",
                "doc_002",
            ],
            "chunk_id": [
                "doc_001_chunk_0000",
                "doc_002_chunk_0000",
                "doc_002_chunk_0001",
            ],
            "title": [
                "Document 1",
                "Old Document 2",
                "Old Document 2",
            ],
            "chunk_text": [
                "Doc 1 content",
                "Old content A",
                "Old content B",
            ],
        }
    )

    delta = pl.DataFrame(
        {
            "document_id": [
                "doc_002",
            ],
            "change_type": [
                "UPDATED",
            ],
            "retrieval_hash": [
                "new_hash_002",
            ],
            "chunk_id": [
                "doc_002_chunk_0000",
            ],
            "chunk_index": [
                0,
            ],
            "title": [
                "Updated Document 2",
            ],
            "chunk_text": [
                "New content",
            ],
        }
    )

    result = build_updated_corpus(
        existing_corpus=existing,
        delta_mapping=delta,
        deleted_document_ids=set(),
    )

    assert result.height == 2

    assert set(
        result["document_id"].to_list()
    ) == {
        "doc_001",
        "doc_002",
    }

    doc_002 = result.filter(
        pl.col("document_id") == "doc_002"
    )

    # Köhnə 2 chunk getdi,
    # yeni yalnız 1 chunk qaldı.
    assert doc_002.height == 1

    assert (
        doc_002["chunk_text"][0]
        == "New content"
    )


def test_new_document_is_added():

    existing = pl.DataFrame(
        {
            "document_id": [
                "doc_001",
            ],
            "chunk_id": [
                "doc_001_chunk_0000",
            ],
            "title": [
                "Document 1",
            ],
            "chunk_text": [
                "Content 1",
            ],
        }
    )

    delta = pl.DataFrame(
        {
            "document_id": [
                "doc_003",
            ],
            "change_type": [
                "NEW",
            ],
            "retrieval_hash": [
                "hash_003",
            ],
            "chunk_id": [
                "doc_003_chunk_0000",
            ],
            "chunk_index": [
                0,
            ],
            "title": [
                "Document 3",
            ],
            "chunk_text": [
                "Content 3",
            ],
        }
    )

    result = build_updated_corpus(
        existing_corpus=existing,
        delta_mapping=delta,
        deleted_document_ids=set(),
    )

    assert result.height == 2

    assert set(
        result["document_id"].to_list()
    ) == {
        "doc_001",
        "doc_003",
    }


def test_deleted_document_is_removed():

    existing = pl.DataFrame(
        {
            "document_id": [
                "doc_001",
                "doc_002",
            ],
            "chunk_id": [
                "doc_001_chunk_0000",
                "doc_002_chunk_0000",
            ],
            "title": [
                "Document 1",
                "Document 2",
            ],
            "chunk_text": [
                "Content 1",
                "Content 2",
            ],
        }
    )

    delta = pl.DataFrame(
        {
            "document_id": [
                "doc_003",
            ],
            "change_type": [
                "NEW",
            ],
            "retrieval_hash": [
                "hash_003",
            ],
            "chunk_id": [
                "doc_003_chunk_0000",
            ],
            "chunk_index": [
                0,
            ],
            "title": [
                "Document 3",
            ],
            "chunk_text": [
                "Content 3",
            ],
        }
    )

    result = build_updated_corpus(
        existing_corpus=existing,
        delta_mapping=delta,
        deleted_document_ids={
            "doc_001"
        },
    )

    assert set(
        result["document_id"].to_list()
    ) == {
        "doc_002",
        "doc_003",
    }


def test_updated_document_with_fewer_chunks_leaves_no_stale_chunks():

    existing = pl.DataFrame(
        {
            "document_id": [
                "doc_010",
                "doc_010",
                "doc_010",
            ],
            "chunk_id": [
                "doc_010_chunk_0000",
                "doc_010_chunk_0001",
                "doc_010_chunk_0002",
            ],
            "title": [
                "Old",
                "Old",
                "Old",
            ],
            "chunk_text": [
                "Old A",
                "Old B",
                "Old C",
            ],
        }
    )

    delta = pl.DataFrame(
        {
            "document_id": [
                "doc_010",
            ],
            "change_type": [
                "UPDATED",
            ],
            "retrieval_hash": [
                "new_hash_010",
            ],
            "chunk_id": [
                "doc_010_chunk_0000",
            ],
            "chunk_index": [
                0,
            ],
            "title": [
                "New",
            ],
            "chunk_text": [
                "Only one new chunk",
            ],
        }
    )

    result = build_updated_corpus(
        existing_corpus=existing,
        delta_mapping=delta,
        deleted_document_ids=set(),
    )

    assert result.height == 1

    assert (
        result["chunk_id"].to_list()
        == [
            "doc_010_chunk_0000"
        ]
    )