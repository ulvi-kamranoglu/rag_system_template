from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from retrieval.search.reranker import BgeReranker


@pytest.mark.parametrize('query,current', [('Who is the rector?',True),('2016 rector',False),('UNEC history',False)])
def test_dates_enter_model_and_score_is_not_adjusted(query,current):
    seen=[]
    def score(pairs,normalize,batch_size):
        assert batch_size == 4
        seen.extend(pairs)
        return [.3,.8]
    reranker=BgeReranker.__new__(BgeReranker)
    reranker.date_context=True
    reranker.model=SimpleNamespace(compute_score=score)
    candidates=[{'chunk_id':'old','payload':{'source_type':'post','title':'Old','chunk_text':'Evidence','date':'2010-01-01','modified':'2026-01-01'}},
                {'chunk_id':'new','payload':{'source_type':'research','title':'New','chunk_text':'Evidence','date':'2010-01-01','modified':'2026-01-01'}}]
    result=reranker.rerank(query,candidates)
    assert '2010-01-01' in seen[0][1] and '2026-01-01' in seen[1][1]
    assert ('Current information as of' in seen[0][0]) is current
    assert result[0]['reranker_score']==.8 and result[1]['reranker_score']==.3
    assert 'reranker_date_context' not in candidates[0]


def test_disabled_context_keeps_original_pair():
    seen=[]
    reranker=BgeReranker.__new__(BgeReranker)
    reranker.date_context=False
    reranker.model=SimpleNamespace(compute_score=lambda pairs,**kw: seen.extend(pairs) or .5)
    payload={'title':'Title','chunk_text':'Content','date':'2026-01-01'}
    reranker.rerank('Question',[{'payload':payload}])
    assert seen==[['Question',BgeReranker.build_passage(payload)]]
