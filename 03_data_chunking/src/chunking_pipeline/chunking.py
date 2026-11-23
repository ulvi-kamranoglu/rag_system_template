"""Paragraph- and sentence-aware chunking with a strict embedding token limit."""
import re
import hashlib
import polars as pl


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False) if text else []


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def sliding_windows(token_ids, budget, overlap):
    """Legacy helper retained for compatibility; not used by boundary-aware chunking."""
    if budget <= 0 or not 0 <= overlap < budget:
        raise ValueError("Invalid chunk budget or overlap")
    if not token_ids:
        return []
    windows, start = [], 0
    while start < len(token_ids):
        end = min(start + budget, len(token_ids))
        windows.append(token_ids[start:end])
        if end == len(token_ids):
            break
        start += budget - overlap
    return windows


def split_sentences(text):
    """Keep normal sentence punctuation attached to its sentence."""
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]


def split_units(text):
    """Return (text, paragraph_index) for each sentence, preserving paragraph boundaries."""
    units = []
    for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n+", text.strip())):
        if paragraph.strip():
            units.extend((sentence, paragraph_index) for sentence in split_sentences(paragraph.strip()))
    return units


def join_units(units):
    if not units:
        return ""
    pieces = [units[0][0]]
    for (text, paragraph), (_, previous_paragraph) in zip(units[1:], units[:-1]):
        pieces.append(("\n\n" if paragraph != previous_paragraph else " ") + text)
    return "".join(pieces)


def split_long_sentence(text, tokenizer, prefix, max_tokens):
    """Split only an oversized single sentence; prefer word boundaries, never discard text."""
    remaining = text
    pieces = []
    while remaining:
        if token_count(tokenizer, prefix + remaining) <= max_tokens:
            pieces.append(remaining)
            break
        left, right, best = 1, len(remaining), 0
        while left <= right:
            mid = (left + right) // 2
            if token_count(tokenizer, prefix + remaining[:mid]) <= max_tokens:
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        if not best:
            raise ValueError("Prefix leaves no room for a single character")
        boundary = max(remaining.rfind(" ", 0, best + 1), remaining.rfind("\n", 0, best + 1))
        # Do not create tiny chunks just to avoid a word split.
        if boundary > best // 2:
            best = boundary + 1
        pieces.append(remaining[:best].strip())
        remaining = remaining[best:].lstrip()
    return [piece for piece in pieces if piece]


def boundary_aware_chunks(text, tokenizer, prefix, max_tokens, overlap_tokens):
    """Pack sentences, preserve paragraphs, and overlap on whole sentences where possible."""
    units = []
    for sentence, paragraph in split_units(text):
        for segment in split_long_sentence(sentence, tokenizer, prefix, max_tokens):
            units.append((segment, paragraph))
    if not units:
        return []

    output = []
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units) and token_count(tokenizer, prefix + join_units(units[start:end + 1])) <= max_tokens:
            end += 1
        current = join_units(units[start:end])
        if token_count(tokenizer, prefix + current) > max_tokens:
            raise ValueError("Chunk exceeds maximum embedding token limit")
        output.append(current)
        if end == len(units):
            break
        # Reuse only whole trailing units. Always consume at least one NEW unit.
        next_start = end
        while next_start > start + 1:
            candidate = next_start - 1
            overlap_text = join_units(units[candidate:end])
            if len(encode(tokenizer, overlap_text)) > overlap_tokens:
                break
            next_start = candidate
        start = next_start
    return output


def build_chunks(df: pl.DataFrame, tokenizer, max_tokens: int, overlap_tokens: int,
                 safety_margin: int = 4, min_content_budget: int = 32) -> pl.DataFrame:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError('Configured overlap must be nonnegative and less than max_tokens')
    records = []
    special = tokenizer.num_special_tokens_to_add(pair=False)
    for row in df.filter(pl.col("retrieval_eligible")).iter_rows(named=True):
        title, content = row["clean_title"], row["clean_content"]
        same = row["title_equals_content"]
        source_text = (title or content) if same else content
        prefix = "" if same or not title else f"Title: {title}\n\nContent:\n"
        budget = max_tokens - len(encode(tokenizer, prefix)) - special - safety_margin
        if budget < min_content_budget and prefix:
            # A title can itself exceed the embedding window. Chunk it together
            # with the body instead of dropping text or repeating an oversized
            # prefix in every chunk. The original title metadata stays intact.
            prefix = ''
            source_text = title + '\n\n' + content
            budget = max_tokens - special - safety_margin
        if budget < min_content_budget:
            raise ValueError(f"Content budget too small for source ID {row['id']!r}: {budget}")
        effective_overlap = overlap_tokens if overlap_tokens < budget else budget // 2
        # Reserve safety margin even though every final text is validated exactly.
        effective_limit = max_tokens - safety_margin
        chunks = boundary_aware_chunks(source_text, tokenizer, prefix, effective_limit, effective_overlap)

        for idx, chunk_text in enumerate(chunks):
            embedding_text = prefix + chunk_text
            count = token_count(tokenizer, embedding_text)
            if count > max_tokens:
                raise ValueError("Embedding token limit exceeded")
            source_type = str(row["source_type"])
            original_id = str(row["id"])
            
            document_id = (
            original_id
            if original_id.startswith(f"{source_type}:")
                else f"{source_type}:{original_id}"
            )
            
            chunk_id = f"{document_id}:chunk:{idx:04d}"   
            
            record = {
                "document_id": document_id,
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "retrieval_hash": hashlib.sha256(
                    (title + "\n\n" + content).encode("utf-8")
                ).hexdigest(),
                "chunk_id": chunk_id,                   # new
                "original_id": original_id,
                "source_type": source_type,
                "chunk_index": idx,
                "title": row.get("title"),
                "chunk_text": chunk_text,
                "embedding_text": embedding_text,
                "chunk_content_tokens": len(encode(tokenizer, chunk_text)),
                "embedding_tokens": count,
                "trimmed_tokens": 0,
                "max_tokens_config": max_tokens,
                "overlap_tokens_config": overlap_tokens,
                "overlap_tokens_effective": effective_overlap,
            }
            for field in ("url", "date", "modified", "source", "language", "metadata", "search_text", "change_type"):
                if field in row:
                    record[field] = row[field]
            records.append(record)
    if not records:
        raise ValueError("No eligible nonempty chunks were generated")
    # Optional metadata can be null for the first hundreds of chunks.
    passthrough = ("title", "url", "date", "modified", "source", "language",
                   "metadata", "search_text", "change_type")
    schema = {name: df.schema[name] for name in passthrough
              if name in df.schema and name in records[0]}
    result = pl.DataFrame(records, schema_overrides=schema)
    if result["chunk_id"].n_unique() != result.height:
        raise ValueError("Duplicate chunk IDs: check uniqueness of (source_type, id) in input")
    return result
