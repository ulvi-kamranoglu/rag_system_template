import json
import pandas as pd
import pytest
from canonical_pipeline.identity import canonicalize_row, normalize_url


def test_wordpress_id():
    row = pd.Series({"id": "60346", "source_type": "page", "metadata": '{"wp_id":"60346"}', "url": "https://UNEC.edu.az/about/?utm_source=test", "language": "az"})
    result = canonicalize_row(row)
    assert result["id"] == "page:60346"
    assert result["url"] == "https://unec.edu.az/about/"


@pytest.mark.parametrize("identifier,kind,source,expected", [
    ('research:001', 'instagram', 'instagram:adalat.muradov', 'instagram:adalat.muradov:1'),
    ('instagram:1', 'instagram', 'instagram:uneceduaz', 'instagram:uneceduaz:1'),
    ('instagram:adalat.muradov:5', 'instagram', 'instagram:adalat.muradov', 'instagram:adalat.muradov:5'),
    ('research:8', 'news', 'website:news.unec.edu.az', 'news:8'),
    ('external:post:3', 'posts', 'unec.edu.az', 'post:3'),
    ('post:3', 'page', 'unec.edu.az', 'page:3'),
])
def test_ids_follow_source_policy(identifier, kind, source, expected):
    row = pd.Series({"id": identifier, "source_type": kind, "source": source, "metadata": '{}',
                     "url": "", "language": "az"})
    result = canonicalize_row(row)
    assert result["id"] == expected
    assert result["source_type"] == ("post" if kind == "posts" else kind)
    assert canonicalize_row(result.copy())['id'] == expected


@pytest.mark.parametrize('identifier', ['research:abc', 'research:', '0', '-1'])
def test_invalid_number_rejected(identifier):
    row = pd.Series(dict(id=identifier, source_type='news', source='website:news.unec.edu.az', metadata='{}', url='', language='az'))
    with pytest.raises(ValueError, match='positive integer'):
        canonicalize_row(row)


def test_instagram_requires_source():
    row = pd.Series(dict(id='research:1', source_type='instagram', source='', metadata='{}', url='', language='az'))
    with pytest.raises(ValueError, match='source prefix'):
        canonicalize_row(row)


def test_wordpress_metadata_conflict_still_rejected():
    row = pd.Series(dict(id='page:3', source_type='posts', metadata='{"wp_id":"99"}', url='', language='az'))
    with pytest.raises(ValueError, match='wp_id conflicts'):
        canonicalize_row(row)


def legal(identifier, metadata='{}'):
    return pd.Series(dict(id=identifier, source_type='legal', source='e-qanun.az', metadata=metadata, url='', language='az'))


@pytest.mark.parametrize('identifier,expected', [
    ('legal:43163:1.1', 'legal:43163:1.1'),
    ('legal:43163:1.2.3.4', 'legal:43163:1.2.3.4'),
    ('legal:27030:4.2-1', 'legal:27030:4.2-1'),
    ('legal:27030:7', 'legal:27030:7'),
    ('research:027030:04.02', 'legal:27030:4.2'),
    ('27030:3', 'legal:27030:3'),
])
def test_legal_ids_keep_framework_and_clause(identifier, expected):
    result = canonicalize_row(legal(identifier))
    assert result['id'] == expected
    assert json.loads(result['metadata'])['clause_id'] == expected.rsplit(':', 1)[1]
    assert canonicalize_row(result.copy())['id'] == expected


def test_same_clause_in_different_frameworks_stays_distinct():
    first = canonicalize_row(legal('legal:43163:7'))['id']
    second = canonicalize_row(legal('legal:27030:7'))['id']
    assert first != second


@pytest.mark.parametrize('identifier', ['legal:7', '7', 'legal:abc:1.1', 'legal:0:1.1', 'legal:43163:', 'legal:43163:1..2', 'legal:43163:1.a'])
def test_invalid_legal_ids_rejected(identifier):
    with pytest.raises(ValueError, match='Legal'):
        canonicalize_row(legal(identifier))


def test_legal_clause_metadata_conflict_rejected():
    with pytest.raises(ValueError, match='clause_id conflicts'):
        canonicalize_row(legal('legal:43163:1.1', '{"clause_id":"1.2"}'))
    assert canonicalize_row(legal('legal:43163:1.1', '{"clause_id":"01.1"}'))['id'] == 'legal:43163:1.1'


def legal_fields(identifier, clause, framework_url='https://e-qanun.az/framework/43163'):
    return pd.Series(dict(id=identifier, source_type='legal', source='e-qanun.az', language='az',
                          url=framework_url, metadata=json.dumps({'clause_id': clause})))


@pytest.mark.parametrize('identifier', ['legal:7', 'x', '', 'legal:43163:7', 'research:43163:07'])
def test_legal_id_is_rebuilt_from_url_and_clause(identifier):
    assert canonicalize_row(legal_fields(identifier, '7'))['id'] == 'legal:43163:7'


def test_legal_url_framework_conflict_rejected():
    with pytest.raises(ValueError, match='framework in URL conflicts'):
        canonicalize_row(legal_fields('legal:27030:7', '7'))


def test_invalid_legal_clause_metadata_rejected():
    with pytest.raises(ValueError, match='clause_id must be numeric'):
        canonicalize_row(legal_fields('legal:43163:7', 'yeddi'))


def wordpress(identifier, kind, wp_id):
    return pd.Series(dict(id=identifier, source_type=kind, source='unec.edu.az', language='az', url='',
                          metadata=json.dumps({'wp_id': wp_id})))


@pytest.mark.parametrize('identifier,kind,wp_id,expected', [
    ('page:60346', 'page', 60346, 'page:60346'),
    ('', 'page', '60346', 'page:60346'),
    ('slug-without-number', 'posts', 493652, 'post:493652'),
    ('content:491051', 'content_post', 491051, 'content_post:491051'),
    ('content:491051', 'content_post', '0491051', 'content_post:491051'),
])
def test_wordpress_id_is_rebuilt_from_wp_id(identifier, kind, wp_id, expected):
    result = canonicalize_row(wordpress(identifier, kind, wp_id))
    assert result['id'] == expected
    assert json.loads(result['metadata'])['wp_id'] == expected.rsplit(':', 1)[1]


def test_invalid_wp_id_rejected():
    with pytest.raises(ValueError, match='wp_id must be a positive integer'):
        canonicalize_row(wordpress('page:3', 'page', 'abc'))


def test_url_preserves_meaningful_queries():
    assert normalize_url("https://example.com/?id=5&utm_medium=x") == "https://example.com/?id=5"
