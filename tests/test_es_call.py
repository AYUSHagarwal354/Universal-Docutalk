"""
Tests for es_call.py (EsConnector)

Covers:
  - data_by_page_no: hits found → merged_text, no hits → None, drive index unless told otherwise
  - search_attachments: only the given fIds in the attachment index, nothing about the user, no thresholds,
    a keyword search is kept for every query and a repeated query is searched once
  - get_conversation_by_id: with history, no conversation, exception, empty qa_pairs
  - store_chat_history: success → True, exception → False
  - delete_conversation: success → True, exception → False
  - create_conversation_index: already exists, creates new, raises on failure
  - search_docs_gpt: routes to hybrid (default) or vector mode, unknown scope → []
  - search_docs_gpt_split_query: routes to hybrid or vector mode
  - rare_word_retriever: only words in at most FUZZY_RARE_MAX of the user's chunks are matched fuzzily,
    within the search filter; switched off, a failed check or no rare word → None
  - search_docs_gpt_split_query (scope username): the fuzzy retriever follows each query's kNN + BM25 pair,
    rare words are looked up once per distinct query; shared scopes are unchanged
  - keyword_text: stop words and single characters are removed, acronyms, numbers and codes are kept
  - keyword searches (multi and single document, every scope) get the trimmed question with minimum_should_match,
    the vector search gets the full question, and the keyword search is skipped when no keyword is left
"""

import pytest
from unittest.mock import MagicMock, patch

from es_call import ATTACHMENT_INDEX, DOCUTALK_INDEX, EsConnector, keyword_text


@pytest.fixture
def connector():
    """Return an EsConnector with its ES client replaced by a MagicMock."""
    with patch("es_call.Elasticsearch"):
        conn = EsConnector()
    conn.es_client = MagicMock()
    return conn


# ─── data_by_page_no ─────────────────────────────────────────────────────────

class TestDataByPageNo:

    def test_returns_merged_text_when_hits_found(self, connector):
        connector.es_client.search.return_value = {
            "hits": {"hits": [
                {"fields": {"text": ["first chunk"]}},
                {"fields": {"text": ["second chunk"]}},
            ]}
        }
        result = connector.data_by_page_no(3, "file_001")
        assert result == {"merged_text": "first chunk second chunk"}

    def test_returns_single_text_for_one_hit(self, connector):
        connector.es_client.search.return_value = {
            "hits": {"hits": [{"fields": {"text": ["only chunk"]}}]}
        }
        result = connector.data_by_page_no(1, "file_002")
        assert result == {"merged_text": "only chunk"}

    def test_returns_none_when_no_hits(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        result = connector.data_by_page_no(99, "file_001")
        assert result is None

    def test_searches_the_drive_unless_told_otherwise(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        connector.data_by_page_no(1, "file_001")
        assert connector.es_client.search.call_args.kwargs["index"] == DOCUTALK_INDEX
        connector.data_by_page_no(1, "file_001", ATTACHMENT_INDEX)
        assert connector.es_client.search.call_args.kwargs["index"] == ATTACHMENT_INDEX


# ─── search_attachments ──────────────────────────────────────────────────────

class TestSearchAttachments:

    def _search(self, connector, queries, fids=("abc", "def")):
        hits = [{"_id": "abc_1_0"}]
        connector.es_client.search.return_value = {"hits": {"hits": hits}}
        assert connector.search_attachments(list(queries), list(fids)) == hits
        return connector.es_client.search.call_args.kwargs

    def test_searches_only_the_given_files_in_the_attachment_index(self, connector):
        call = self._search(connector, ["what is the leave policy"])
        assert call["index"] == ATTACHMENT_INDEX
        assert call.get("routing") is None
        retrievers = call["body"]["retriever"]["rrf"]["retrievers"]
        assert len(retrievers) == 2
        expected = [{"terms": {"fId": ["abc", "def"]}}, {"term": {"in_trash": False}}]
        assert retrievers[0]["knn"]["filter"] == expected
        assert retrievers[1]["standard"]["filter"] == expected

    def test_nothing_about_the_user_or_a_threshold(self, connector):
        retrievers = self._search(connector, ["what is the leave policy"])["body"]["retriever"]["rrf"]["retrievers"]
        text = str(retrievers)
        assert "username" not in text and "path.text" not in text and "shared_with" not in text
        assert "similarity" not in retrievers[0]["knn"]
        assert "min_score" not in retrievers[1]["standard"]
        assert "minimum_should_match" not in text

    def test_vector_gets_the_question_and_keywords_the_trimmed_one(self, connector):
        retrievers = self._search(connector, ["what is the leave policy"])["body"]["retriever"]["rrf"]["retrievers"]
        assert retrievers[0]["knn"]["query_vector_builder"]["text_embedding"]["model_text"] == "what is the leave policy"
        assert retrievers[1]["standard"]["query"]["multi_match"]["query"] == "leave policy"

    def test_a_question_of_only_stop_words_keeps_its_keyword_search(self, connector):
        retrievers = self._search(connector, ["what is this"])["body"]["retriever"]["rrf"]["retrievers"]
        assert len(retrievers) == 2
        assert retrievers[1]["standard"]["query"]["multi_match"]["query"] == "what is this"

    def test_a_repeated_query_is_searched_once(self, connector):
        retrievers = self._search(connector, ["same", "same", "other"])["body"]["retriever"]["rrf"]["retrievers"]
        assert len(retrievers) == 4


# ─── get_conversation_by_id ───────────────────────────────────────────────────

class TestGetConversationById:

    def test_returns_formatted_qa_pairs(self, connector):
        connector.es_client.search.return_value = {
            "hits": {"hits": [{
                "_source": {
                    "qa_pairs": [
                        {"question": "What is Redis?", "answer": "An in-memory DB"},
                        {"question": "What is ES?", "answer": "A search engine"},
                    ]
                }
            }]}
        }
        result = connector.get_conversation_by_id("conv_001")
        assert "What is Redis?" in result
        assert "An in-memory DB" in result
        assert "What is ES?" in result

    def test_returns_none_when_no_document_found(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        result = connector.get_conversation_by_id("conv_missing")
        assert result is None

    def test_returns_none_on_elasticsearch_exception(self, connector):
        connector.es_client.search.side_effect = Exception("cluster unavailable")
        result = connector.get_conversation_by_id("conv_err")
        assert result is None

    def test_returns_empty_string_when_qa_pairs_empty(self, connector):
        connector.es_client.search.return_value = {
            "hits": {"hits": [{"_source": {"qa_pairs": []}}]}
        }
        result = connector.get_conversation_by_id("conv_empty")
        assert result == ""

    def test_respects_history_size_limit(self, connector):
        # HISTORY_SIZE defaults to 3, so only last 3 pairs returned
        pairs = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)
        ]
        connector.es_client.search.return_value = {
            "hits": {"hits": [{"_source": {"qa_pairs": pairs}}]}
        }
        result = connector.get_conversation_by_id("conv_long")
        # Only last 3 questions should appear
        assert "Q9" in result
        assert "Q0" not in result


# ─── store_chat_history ───────────────────────────────────────────────────────

class TestStoreChatHistory:

    def test_returns_true_on_successful_update(self, connector):
        connector.es_client.update.return_value = {"result": "updated"}
        result = connector.store_chat_history("conv_1", "What is Redis?", "An in-memory DB")
        assert result is True

    def test_calls_es_update_with_correct_index_and_id(self, connector):
        connector.es_client.update.return_value = {}
        connector.store_chat_history("conv_42", "q", "a")
        call_kwargs = connector.es_client.update.call_args[1]
        assert call_kwargs["id"] == "conv_42"

    def test_returns_false_on_exception(self, connector):
        connector.es_client.update.side_effect = Exception("update failed")
        result = connector.store_chat_history("conv_1", "question", "answer")
        assert result is False


# ─── delete_conversation ─────────────────────────────────────────────────────

class TestDeleteConversation:

    def test_returns_true_on_successful_delete(self, connector):
        connector.es_client.delete.return_value = {"result": "deleted"}
        result = connector.delete_conversation("conv_del")
        assert result is True

    def test_calls_es_delete_with_correct_id(self, connector):
        connector.es_client.delete.return_value = {}
        connector.delete_conversation("my_conv_id")
        call_kwargs = connector.es_client.delete.call_args[1]
        assert call_kwargs["id"] == "my_conv_id"

    def test_returns_false_on_exception(self, connector):
        connector.es_client.delete.side_effect = Exception("document not found")
        result = connector.delete_conversation("conv_missing")
        assert result is False


# ─── create_conversation_index ───────────────────────────────────────────────

class TestCreateConversationIndex:

    def test_skips_creation_when_index_already_exists(self, connector):
        connector.es_client.indices.exists.return_value = True
        connector.create_conversation_index()
        connector.es_client.indices.create.assert_not_called()

    def test_creates_index_when_it_does_not_exist(self, connector):
        connector.es_client.indices.exists.return_value = False
        connector.es_client.indices.create.return_value = {"acknowledged": True}
        connector.create_conversation_index()
        connector.es_client.indices.create.assert_called_once()

    def test_raises_exception_on_index_creation_failure(self, connector):
        connector.es_client.indices.exists.return_value = False
        connector.es_client.indices.create.side_effect = Exception("cluster error")
        with pytest.raises(Exception, match="cluster error"):
            connector.create_conversation_index()


# ─── search_docs_gpt ─────────────────────────────────────────────────────────

class TestSearchDocsGpt:

    def test_routes_to_hybrid_rrf_by_default(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": [{"_id": "doc1"}]}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            result = connector.search_docs_gpt("query", "user1", "T_path", "username")
        assert result == [{"_id": "doc1"}]

    def test_routes_to_vector_when_retriever_mode_is_vector(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": [{"_id": "doc2"}]}}
        with patch("es_call.RETRIEVER_MODE", "vector"):
            result = connector.search_docs_gpt("query", "user1", "T_path", "username")
        assert result == [{"_id": "doc2"}]

    def test_returns_empty_list_for_unknown_scope_in_hybrid(self, connector):
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            result = connector.search_docs_gpt("query", "user1", "T_path", "unknown_scope")
        assert result == []

    def test_returns_empty_list_for_unknown_scope_in_vector(self, connector):
        with patch("es_call.RETRIEVER_MODE", "vector"):
            result = connector.search_docs_gpt("query", "user1", "T_path", "unknown_scope")
        assert result == []

    def test_shared_with_scope_uses_shared_with_filter(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            connector.search_docs_gpt("query", "user1", "T_path", "shared_with")
        body = connector.es_client.search.call_args[1]["body"]
        # For shared_with scope routing should be None
        assert connector.es_client.search.call_args[1]["routing"] is None


# ─── search_docs_gpt_split_query ─────────────────────────────────────────────

class TestSearchDocsGptSplitQuery:

    def test_routes_to_hybrid_by_default(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": [{"_id": "hit1"}]}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            result = connector.search_docs_gpt_split_query(
                ["q1", "q2"], "user1", "T_path", "username"
            )
        assert result == [{"_id": "hit1"}]

    def test_routes_to_vector_when_configured(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": [{"_id": "hit2"}]}}
        with patch("es_call.RETRIEVER_MODE", "vector"):
            result = connector.search_docs_gpt_split_query(
                ["q1"], "user1", "T_path", "username"
            )
        assert result == [{"_id": "hit2"}]

    def test_returns_empty_list_for_unknown_scope(self, connector):
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            result = connector.search_docs_gpt_split_query(
                ["q1"], "user1", "T_path", "invalid_scope"
            )
        assert result == []


# ─── rare_word_retriever (misspelt words) ─────────────────────────────────────

FILTERS = [{"term": {"username": "user1"}}]


def _analyzed(*words):
    return {"tokens": [{"token": word} for word in words]}


def _user_chunk_counts(*counts):
    return {"aggregations": {"user": {"t": {"buckets": {f"t{i}": {"doc_count": n} for i, n in enumerate(counts)}}}}}


def _first_keys(retrievers):
    return [next(iter(retriever)) for retriever in retrievers]


class TestRareWordRetriever:

    def test_matches_only_the_rare_words_fuzzily(self, connector):
        # "who" and "is" are in many of the user's chunks, "d" in one, the misspelt "gosh" in none
        connector.es_client.indices.analyze.return_value = _analyzed("who", "is", "d", "gosh")
        connector.es_client.search.return_value = _user_chunk_counts(90, 1289, 1, 0)
        with patch("es_call.FUZZY_RARE_MAX", 1):
            retriever = connector.rare_word_retriever("who is d. gosh?", FILTERS, "user1_0")
        should = retriever["standard"]["query"]["bool"]["should"]
        assert [match["match"]["text"]["query"] for match in should] == ["d", "gosh"]
        assert should[1]["match"]["text"] == {"query": "gosh", "fuzziness": "AUTO", "prefix_length": 1, "max_expansions": 50}
        assert retriever["standard"]["filter"] == FILTERS

    def test_counts_the_words_within_the_search_filter_and_routing(self, connector):
        connector.es_client.indices.analyze.return_value = _analyzed("gosh")
        connector.es_client.search.return_value = _user_chunk_counts(0)
        connector.rare_word_retriever("gosh", FILTERS, "user1_0")
        call = connector.es_client.search.call_args[1]
        assert call["routing"] == "user1_0"
        assert call["body"]["aggs"]["user"]["filter"] == {"bool": {"filter": FILTERS}}

    def test_returns_none_when_every_word_is_common(self, connector):
        connector.es_client.indices.analyze.return_value = _analyzed("who", "is", "ghosh")
        connector.es_client.search.return_value = _user_chunk_counts(90, 1289, 3)
        with patch("es_call.FUZZY_RARE_MAX", 1):
            assert connector.rare_word_retriever("who is ghosh", FILTERS, "user1_0") is None

    def test_returns_none_without_any_word(self, connector):
        connector.es_client.indices.analyze.return_value = _analyzed()
        assert connector.rare_word_retriever("?", FILTERS, "user1_0") is None
        connector.es_client.search.assert_not_called()

    def test_switched_off_makes_no_elasticsearch_call(self, connector):
        with patch("es_call.FUZZY_RARE_WORDS", False):
            assert connector.rare_word_retriever("who is d. gosh?", FILTERS, "user1_0") is None
        connector.es_client.indices.analyze.assert_not_called()
        connector.es_client.search.assert_not_called()

    def test_failed_check_returns_none(self, connector):
        connector.es_client.indices.analyze.side_effect = Exception("cluster unavailable")
        assert connector.rare_word_retriever("who is d. gosh?", FILTERS, "user1_0") is None


class TestSplitQueryRareWords:

    def test_fuzzy_retriever_follows_each_query_pair(self, connector):
        fuzzy = {"standard": {"query": "fuzzy"}}
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"), \
             patch.object(connector, "rare_word_retriever", return_value=fuzzy) as rare:
            connector.search_docs_gpt_split_query(["who is d. gosh?", "who is d. gosh?"], "user1", "T_", "username")
        retrievers = connector.es_client.search.call_args[1]["body"]["retriever"]["rrf"]["retrievers"]
        assert _first_keys(retrievers) == ["knn", "standard", "standard", "knn", "standard", "standard"]
        assert retrievers[2] == fuzzy and retrievers[5] == fuzzy
        # looked up once for the two identical query strings, with the BM25 filter and the user's routing,
        # on the question without stop words and single letters
        rare.assert_called_once()
        query, filters, route = rare.call_args[0]
        assert query == "gosh"
        assert filters == retrievers[1]["standard"]["filter"]
        assert route == "user1_0,user1_1,user1_2"

    def test_no_extra_retriever_without_rare_words(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"), \
             patch.object(connector, "rare_word_retriever", return_value=None):
            connector.search_docs_gpt_split_query(["who is d. ghosh?"], "user1", "T_", "username")
        retrievers = connector.es_client.search.call_args[1]["body"]["retriever"]["rrf"]["retrievers"]
        assert _first_keys(retrievers) == ["knn", "standard"]

    def test_shared_scopes_are_unchanged(self, connector):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"), \
             patch.object(connector, "rare_word_retriever") as rare:
            connector.search_docs_gpt_split_query(["q"], "user1", "T_", "shared_with")
            connector.search_docs_gpt_split_query(["q"], "user1", "T_", "shared_by")
        rare.assert_not_called()


# ─── keyword_text and the keyword searches ────────────────────────────────────

class TestKeywordText:

    @pytest.mark.parametrize("question, keywords", [
        ("who is d. gosh?", "gosh"),
        ("What is the PLA Navy", "PLA Navy"),
        ("MDT Gas sample Analysis", "MDT Gas sample Analysis"),
        ("Requirement analysis using AI", "Requirement analysis using AI"),
        ("When was PML status awarded to the block?", "PML status awarded block"),
        ("Type 093 submarine", "Type 093 submarine"),
        ("Objective of location BGK-42-J (GK-28 PML)", "Objective location BGK-42-J GK-28 PML"),
        ("What is Ghosh's designation?", "Ghosh's designation"),
        # an initial written without a space is dropped too; abbreviations and decimals are kept
        ("Who is d.gosh?", "gosh"),
        ("Who is D.Ghosh", "Ghosh"),
        ("U.S. Navy ships", "U.S Navy ships"),
        ("salinity of 9.2ppg brine", "salinity 9.2ppg brine"),
    ])
    def test_keeps_only_meaningful_words(self, question, keywords):
        assert keyword_text(question) == keywords

    def test_nothing_left_returns_empty_text(self):
        assert keyword_text("Who is it?") == ""
        assert keyword_text("?") == ""


def _rrf_retrievers(connector):
    return connector.es_client.search.call_args[1]["body"]["retriever"]["rrf"]["retrievers"]


class TestKeywordSearches:

    @pytest.mark.parametrize("scope", ["username", "shared_with", "shared_by"])
    def test_split_query_keyword_search_gets_trimmed_question(self, connector, scope):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"),              patch.object(connector, "rare_word_retriever", return_value=None):
            connector.search_docs_gpt_split_query(["What is the PLA Navy?"], "user1", "T_", scope)
        knn, keyword = _rrf_retrievers(connector)
        # the vector search keeps the full question
        assert knn["knn"]["query_vector_builder"]["text_embedding"]["model_text"] == "What is the PLA Navy?"
        assert keyword["standard"]["query"]["multi_match"] == {"query": "PLA Navy", "fields": ["text"], "minimum_should_match": "75%"}

    @pytest.mark.parametrize("scope", ["username", "shared_with", "shared_by"])
    def test_split_query_skips_keyword_search_when_no_keyword_is_left(self, connector, scope):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"),              patch.object(connector, "rare_word_retriever", return_value=None):
            connector.search_docs_gpt_split_query(["Who is it?", "who is it"], "user1", "T_", scope)
        assert _first_keys(_rrf_retrievers(connector)) == ["knn", "knn"]

    @pytest.mark.parametrize("scope", ["username", "shared_with", "shared_by"])
    def test_single_document_keyword_search_gets_trimmed_question(self, connector, scope):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            connector.search_docs_gpt("Who is D. Ghosh?", "user1", "T_", scope)
        keyword, knn = _rrf_retrievers(connector)
        assert keyword["standard"]["query"]["match"]["text"] == {"query": "Ghosh", "minimum_should_match": "75%"}
        assert knn["knn"]["query_vector_builder"]["text_embedding"]["model_text"] == "Who is D. Ghosh?"

    @pytest.mark.parametrize("scope", ["username", "shared_with", "shared_by"])
    def test_single_document_skips_keyword_search_when_no_keyword_is_left(self, connector, scope):
        connector.es_client.search.return_value = {"hits": {"hits": []}}
        with patch("es_call.RETRIEVER_MODE", "hybrid_rrf"):
            connector.search_docs_gpt("Who is it?", "user1", "T_", scope)
        assert _first_keys(_rrf_retrievers(connector)) == ["knn"]

    def test_fuzzy_check_is_skipped_without_keywords(self, connector):
        assert connector.rare_word_retriever("", FILTERS, "user1_0") is None
        connector.es_client.indices.analyze.assert_not_called()
