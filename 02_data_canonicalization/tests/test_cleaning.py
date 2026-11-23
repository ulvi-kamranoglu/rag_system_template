from canonical_pipeline.cleaning import clean_text


def test_paragraphs_and_entities():
    assert clean_text("<p>Hello&nbsp;world</p><p>Second line</p>") == "Hello world\n\nSecond line"


def test_removes_scripts():
    assert clean_text("<script>bad()</script><p>Good</p>") == "Good"


def test_idempotent():
    sample = "<p>First</p><p>Second</p>"
    assert clean_text(clean_text(sample)) == clean_text(sample)
