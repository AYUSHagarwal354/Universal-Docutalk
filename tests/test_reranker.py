"""
Tests for reranker.py

Covers:
  rerank_hits:
    - hits above the floor are returned best first with a 0-1 score, marked relevant
    - hits far below the best chunk are kept when they are above the floor
    - hits below the floor are dropped
    - nothing above the floor returns the best RERANK_MIN_KEEP hits marked not relevant, or [] when RERANK_MIN_KEEP is 0
    - at most RERANK_KEEP hits are returned
    - only the first RERANK_CANDIDATES clean hits are sent to the reranker
    - short and repeated chunks never reach the reranker
    - API failure or a disabled reranker keeps the elasticsearch order, marked relevant
    - empty hits, or only short / repeated hits, make no API call
    - raw watsonx scores from real requests: a well matched, a weakly related and an off-topic question

  clean_hits:
    - drops chunks shorter than RERANK_MIN_CHUNK_CHARS after removing [CLS]/[SEP]
    - drops chunks whose text repeats an earlier chunk, ignoring case, spacing and [CLS]/[SEP]

  score_texts:
    - sends the query and texts to the rerank API
    - maps scores back to the input order
    - a text missing from the response gets the lowest score
    - HTTP error or token failure returns None
"""

import pytest
from unittest.mock import MagicMock, patch
import reranker


@pytest.fixture(autouse=True)
def fake_token():
    # the IAM mock in conftest is shared by every test file, so the token is patched here
    with patch("reranker.get_token", return_value="Bearer fake-test-token"):
        yield


@pytest.fixture(autouse=True)
def fixed_settings():
    # independent of the defaults and the environment
    with patch.multiple(reranker, RERANK_ENABLED=True, RERANK_SCORE_FLOOR=-8.0, RERANK_CANDIDATES=20,
                        RERANK_KEEP=10, RERANK_MIN_KEEP=5, RERANK_MIN_CHUNK_CHARS=20):
        yield


def _hit(fid, hit_id, text=None):
    # by default every hit gets its own text, long enough to pass clean_hits
    text = text if text is not None else f"passage {hit_id} of file {fid} with enough words"
    return {"_score": 0.03, "_id": hit_id, "_source": {"fId": fid, "fileName": f"{fid}.pdf", "text": text}}


def _api_response(scores):
    response = MagicMock()
    response.json.return_value = {"results": [{"index": i, "score": s} for i, s in enumerate(scores)]}
    return response


class TestRerankHits:

    def test_keeps_hits_above_floor_best_first_with_0_1_score(self):
        hits = [_hit("f1", "a"), _hit("f2", "b"), _hit("f3", "c")]
        with patch("reranker.requests.post", return_value=_api_response([-9.0, 2.0, 1.0])):
            kept, relevant = reranker.rerank_hits("q", hits)
        assert [h["_id"] for h in kept] == ["b", "c"]
        assert relevant is True
        # raw 2.0 and 1.0 are reported as 0-1 scores
        assert [h["_score"] for h in kept] == pytest.approx([0.881, 0.731], abs=0.001)

    def test_hits_far_below_best_are_kept_when_above_floor(self):
        # a strong best chunk (10.0) no longer pushes out related chunks: there is no margin, only the floor
        with patch("reranker.requests.post", return_value=_api_response([10.0, 4.5, -5.5])):
            kept, relevant = reranker.rerank_hits("q", [_hit("f1", "a"), _hit("f2", "b"), _hit("f3", "c")])
        assert [h["_id"] for h in kept] == ["a", "b", "c"]

    def test_hits_below_floor_are_dropped(self):
        with patch("reranker.requests.post", return_value=_api_response([-5.0, -7.5, -8.5])):
            kept, relevant = reranker.rerank_hits("q", [_hit("f1", "a"), _hit("f2", "b"), _hit("f3", "c")])
        assert [h["_id"] for h in kept] == ["a", "b"]
        assert relevant is True

    def test_nothing_above_floor_returns_best_hits_not_relevant(self):
        hits = [_hit(f"f{i}", str(i)) for i in range(7)]
        with patch("reranker.requests.post", return_value=_api_response([-11.0, -9.0, -10.0, -9.5, -12.0, -10.5, -11.5])):
            kept, relevant = reranker.rerank_hits("q", hits)
        # the best 5 by reranker score, for the model only
        assert [h["_id"] for h in kept] == ["1", "3", "2", "5", "0"]
        assert relevant is False

    def test_nothing_above_floor_returns_nothing_when_min_keep_is_zero(self):
        with patch.object(reranker, "RERANK_MIN_KEEP", 0), \
             patch("reranker.requests.post", return_value=_api_response([-9.0, -10.0])):
            assert reranker.rerank_hits("q", [_hit("f1", "a"), _hit("f2", "b")]) == ([], False)

    def test_many_chunks_from_one_file_are_all_kept(self):
        hits = [_hit("f1", str(i)) for i in range(5)]
        with patch("reranker.requests.post", return_value=_api_response([2.0] * 5)):
            kept, _ = reranker.rerank_hits("q", hits)
        assert len(kept) == 5

    def test_returns_at_most_rerank_keep_hits(self):
        hits = [_hit(f"f{i}", str(i)) for i in range(15)]
        with patch("reranker.requests.post", return_value=_api_response([2.0] * 15)):
            kept, _ = reranker.rerank_hits("q", hits)
        assert len(kept) == 10

    def test_only_first_candidates_are_sent_to_reranker(self):
        hits = [_hit(f"f{i}", str(i)) for i in range(25)]
        with patch("reranker.requests.post", return_value=_api_response([2.0] * 20)) as mock_post:
            reranker.rerank_hits("q", hits)
        assert len(mock_post.call_args.kwargs["json"]["inputs"]) == 20

    def test_short_and_repeated_chunks_do_not_take_reranker_slots(self):
        hits = [_hit("f1", "a"), _hit("f1", "short", "Top of Form"), _hit("copy", "a-copy", _hit("f1", "a")["_source"]["text"]),
                _hit("f2", "b")]
        with patch("reranker.requests.post", return_value=_api_response([2.0, 1.0])) as mock_post:
            kept, _ = reranker.rerank_hits("q", hits)
        assert len(mock_post.call_args.kwargs["json"]["inputs"]) == 2
        assert [h["_id"] for h in kept] == ["a", "b"]

    def test_api_failure_keeps_elasticsearch_order(self):
        hits = [_hit("f1", "a"), _hit("f2", "b")]
        with patch("reranker.requests.post", side_effect=Exception("timeout")):
            kept, relevant = reranker.rerank_hits("q", hits)
        assert [h["_id"] for h in kept] == ["a", "b"]
        assert relevant is True
        # elasticsearch scores are left untouched
        assert kept[0]["_score"] == 0.03

    def test_disabled_reranker_keeps_elasticsearch_order_without_api_call(self):
        hits = [_hit("f1", "a"), _hit("f2", "b")]
        with patch.object(reranker, "RERANK_ENABLED", False), \
             patch("reranker.requests.post") as mock_post:
            kept, relevant = reranker.rerank_hits("q", hits)
        assert [h["_id"] for h in kept] == ["a", "b"]
        assert relevant is True
        mock_post.assert_not_called()

    def test_empty_hits_returns_empty_without_api_call(self):
        with patch("reranker.requests.post") as mock_post:
            assert reranker.rerank_hits("q", []) == ([], False)
        mock_post.assert_not_called()

    def test_only_short_chunks_returns_empty_without_api_call(self):
        with patch("reranker.requests.post") as mock_post:
            assert reranker.rerank_hits("q", [_hit("f1", "a", "Pakistan"), _hit("f2", "b", "[CLS] Armament [SEP]")]) == ([], False)
        mock_post.assert_not_called()

    def test_raw_watsonx_scores_keep_the_chunks_of_the_relevant_file(self):
        # raw scores returned by watsonx ms-marco-minilm-l-12-v2 for "What are Acreage details":
        # the first 7 chunks came from the relevant report, the other 13 from unrelated files
        raw_scores = [-3.4766, -7.3633, -7.4297, -7.4844, -7.5547, -7.9961, -8.6797, -10.7812, -10.7969, -10.9766,
                      -11.0234, -11.0938, -11.1016, -11.125, -11.1328, -11.1406, -11.1406, -11.1641, -11.2969, -11.3125]
        hits = [_hit("report" if i < 7 else f"other{i}", str(i)) for i in range(len(raw_scores))]
        with patch("reranker.requests.post", return_value=_api_response(raw_scores)):
            kept, relevant = reranker.rerank_hits("q", hits)
        assert [h["_id"] for h in kept] == ["0", "1", "2", "3", "4", "5"]
        assert {h["_source"]["fId"] for h in kept} == {"report"}
        assert relevant is True
        assert kept[0]["_score"] == pytest.approx(0.03, abs=0.001)

    def test_raw_watsonx_scores_of_a_weakly_related_question_are_kept(self):
        # raw scores for "MDT Gas sample Analysis": the best chunk scores low, but the question must be answered
        raw_scores = [-5.2148, -5.4141, -6.2266, -7.7305, -8.5391, -8.6719, -9.3516, -9.4141, -9.9453, -10.5234,
                      -10.5859, -10.6562, -10.8438, -10.8594, -10.9844, -11.0547, -11.0859, -11.1016, -11.1328, -11.1406]
        hits = [_hit("wcr" if i < 9 else f"other{i}", str(i)) for i in range(len(raw_scores))]
        with patch("reranker.requests.post", return_value=_api_response(raw_scores)):
            kept, relevant = reranker.rerank_hits("q", hits)
        assert [h["_id"] for h in kept] == ["0", "1", "2", "3"]
        assert relevant is True

    def test_raw_watsonx_scores_of_an_off_topic_question_are_not_relevant(self):
        # raw scores for "What is the capital of France", the highest best-score seen for an off-topic question
        raw_scores = [-9.09, -9.43, -9.47, -9.48, -9.56, -10.2, -10.9, -11.1]
        hits = [_hit(f"f{i}", str(i)) for i in range(len(raw_scores))]
        with patch("reranker.requests.post", return_value=_api_response(raw_scores)):
            kept, relevant = reranker.rerank_hits("q", hits)
        assert relevant is False


class TestCleanHits:

    def test_drops_chunks_shorter_than_min_chars(self):
        hits = [_hit("f1", "word", "Pakistan"), _hit("f1", "form", "[CLS] Top of Form [SEP]"),
                _hit("f1", "fact", "Succeeded by Type 093"), _hit("f1", "long", "x" * 40)]
        assert [h["_id"] for h in reranker.clean_hits(hits)] == ["fact", "long"]

    def test_drops_repeated_text_ignoring_case_spacing_and_markers(self):
        hits = [_hit("f1", "first", "[CLS] From Wikipedia, the free encyclopedia [SEP]"),
                _hit("f2", "again", "from wikipedia,   the FREE encyclopedia"),
                _hit("f3", "other", "From Wikipedia, the free encyclopedia of ships")]
        assert [h["_id"] for h in reranker.clean_hits(hits)] == ["first", "other"]

    def test_keeps_order_of_the_remaining_hits(self):
        hits = [_hit("f1", "c"), _hit("f2", "a"), _hit("f3", "b")]
        assert [h["_id"] for h in reranker.clean_hits(hits)] == ["c", "a", "b"]


class TestScoreTexts:

    def test_sends_query_and_texts_to_rerank_api(self):
        with patch("reranker.requests.post", return_value=_api_response([0.7, 0.1])) as mock_post:
            reranker.score_texts("my question", ["first", "second"])
        body = mock_post.call_args.kwargs["json"]
        assert body["query"] == "my question"
        assert body["inputs"] == [{"text": "first"}, {"text": "second"}]
        assert body["model_id"] == reranker.RERANK_MODEL_ID

    def test_scores_are_mapped_back_to_input_order(self):
        # the API returns results sorted by score, not by input order
        response = MagicMock()
        response.json.return_value = {"results": [{"index": 1, "score": 1.5}, {"index": 0, "score": -8.2}]}
        with patch("reranker.requests.post", return_value=response):
            assert reranker.score_texts("q", ["first", "second"]) == [-8.2, 1.5]

    def test_text_missing_from_response_gets_lowest_score(self):
        response = MagicMock()
        response.json.return_value = {"results": [{"index": 0, "score": 1.5}]}
        with patch("reranker.requests.post", return_value=response):
            assert reranker.score_texts("q", ["first", "second"]) == [1.5, float("-inf")]

    def test_http_error_returns_none(self):
        response = MagicMock()
        response.raise_for_status.side_effect = Exception("401 Unauthorized")
        with patch("reranker.requests.post", return_value=response):
            assert reranker.score_texts("q", ["first"]) is None

    def test_token_failure_returns_none_without_api_call(self):
        with patch("reranker.get_token", side_effect=Exception("iam unavailable")), \
             patch("reranker.requests.post") as mock_post:
            assert reranker.score_texts("q", ["first"]) is None
        mock_post.assert_not_called()
