"""
Tests for query_enhancement.py

Covers:
  - get_token(): success and RequestException paths
  - ibm_model_query() with IBM model:
      multiDocument success, singleDocument success,
      API exception, JSON parse failure, missing key in response, empty results list
  - ibm_model_query() with VLLM model:
      success and exception paths
"""

import pytest
import requests
from unittest.mock import MagicMock, patch

import query_enhancement
from query_enhancement import get_token, ibm_model_query


# ─── get_token ────────────────────────────────────────────────────────────────

class TestGetToken:

    def test_get_token_returns_bearer_prefix_with_token(self):
        with patch.object(query_enhancement, "authenticator") as mock_auth:
            mock_auth.token_manager.get_token.return_value = "abc123"
            result = get_token()
        assert result == "Bearer abc123"

    def test_get_token_raises_request_exception_on_network_failure(self):
        with patch.object(query_enhancement, "authenticator") as mock_auth:
            mock_auth.token_manager.get_token.side_effect = (
                requests.exceptions.RequestException("network error")
            )
            with pytest.raises(requests.exceptions.RequestException):
                get_token()


# ─── ibm_model_query – IBM model ─────────────────────────────────────────────

class TestIbmModelQueryIBM:

    def _mock_response(self, generated_text):
        resp = MagicMock()
        resp.json.return_value = {"results": [{"generated_text": generated_text}]}
        return resp

    def test_multi_document_returns_generated_text(self):
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post",
                   return_value=self._mock_response("['sub-query 1', 'sub-query 2']")):
            result = ibm_model_query("Compare A and B", "IBM", "chat history", "multiDocument")
        assert "sub-query" in result

    def test_single_document_returns_enhanced_query(self):
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post",
                   return_value=self._mock_response("What are the features of Redis?")):
            result = ibm_model_query("What are its features?", "IBM", "Discussion about Redis", "singleDocument")
        assert result == "What are the features of Redis?"

    def test_api_call_exception_returns_fallback_message(self):
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post", side_effect=Exception("timeout")):
            result = ibm_model_query("query", "IBM", "", "multiDocument")
        assert result == "I am unable to respond right now. Please try again later."

    def test_json_parse_failure_returns_fallback_message(self):
        bad_resp = MagicMock()
        bad_resp.json.side_effect = Exception("not valid json")
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post", return_value=bad_resp):
            result = ibm_model_query("query", "IBM", "", "multiDocument")
        assert result == "I am unable to respond right now. Please try again later."

    def test_missing_results_key_returns_fallback_message(self):
        resp = MagicMock()
        resp.json.return_value = {"error": "model unavailable"}
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post", return_value=resp):
            result = ibm_model_query("query", "IBM", "", "multiDocument")
        assert result == "I am unable to respond right now. Please try again later."

    def test_empty_results_list_returns_fallback_message(self):
        resp = MagicMock()
        resp.json.return_value = {"results": []}
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post", return_value=resp):
            result = ibm_model_query("query", "IBM", "", "multiDocument")
        assert result == "I am unable to respond right now. Please try again later."

    def test_post_called_with_correct_url(self):
        with patch.object(query_enhancement, "get_token", return_value="Bearer tok"), \
             patch("query_enhancement.requests.post",
                   return_value=self._mock_response("result")) as mock_post:
            ibm_model_query("query", "IBM", "", "multiDocument")
        called_url = mock_post.call_args[0][0]
        assert "us-south.ml.cloud.ibm.com" in called_url

    def test_authorization_header_contains_bearer_token(self):
        with patch.object(query_enhancement, "get_token", return_value="Bearer my-token"), \
             patch("query_enhancement.requests.post",
                   return_value=self._mock_response("result")) as mock_post:
            ibm_model_query("query", "IBM", "", "multiDocument")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-token"


# ─── ibm_model_query – VLLM model ────────────────────────────────────────────

class TestIbmModelQueryVLLM:

    def test_vllm_success_returns_model_content(self):
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "VLLM generated answer"}}]
        }
        with patch("query_enhancement.requests.post", return_value=resp):
            result = ibm_model_query("What is Elasticsearch?", "VLLM", "", "multiDocument")
        assert result == "VLLM generated answer"

    def test_vllm_exception_returns_none(self):
        with patch("query_enhancement.requests.post", side_effect=Exception("connection refused")):
            result = ibm_model_query("query", "VLLM", "", "multiDocument")
        assert result is None
