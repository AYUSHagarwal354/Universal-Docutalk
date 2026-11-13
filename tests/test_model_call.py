"""
Tests for model_call.py (ModelCall)

Covers:
  - get_token(): returns "Bearer <token>", raises RequestException on failure
  - vllm_model_call(): success returns response content, exception returns None
  - send_post_request(): yields text chunks and handles moderation events
"""

import pytest
import requests
from unittest.mock import MagicMock, AsyncMock, patch, call

from model_call import ModelCall


@pytest.fixture
def model():
    """Return a ModelCall instance. IAMAuthenticator is already mocked by conftest."""
    return ModelCall()


# ─── get_token ────────────────────────────────────────────────────────────────

class TestGetToken:

    def test_returns_bearer_token_string(self, model):
        model.authenticator.token_manager.get_token.return_value = "test-token-xyz"
        result = model.get_token()
        assert result == "Bearer test-token-xyz"

    def test_raises_request_exception_on_network_failure(self, model):
        model.authenticator.token_manager.get_token.side_effect = (
            requests.exceptions.RequestException("iam unavailable")
        )
        with pytest.raises(requests.exceptions.RequestException):
            model.get_token()


# ─── vllm_model_call ─────────────────────────────────────────────────────────

class TestVllmModelCall:

    def test_returns_message_content_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "VLLM response text"}}]
        }
        with patch("model_call.requests.post", return_value=mock_resp):
            result = ModelCall.vllm_model_call("history", "context text", "question", "NLP")
        assert result == "VLLM response text"

    def test_returns_none_on_connection_exception(self):
        with patch("model_call.requests.post", side_effect=Exception("connection refused")):
            result = ModelCall.vllm_model_call("history", "text", "question", "NLP")
        assert result is None

    def test_returns_none_on_bad_json_response(self):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = Exception("invalid json")
        with patch("model_call.requests.post", return_value=mock_resp):
            result = ModelCall.vllm_model_call("history", "text", "question", "EXT")
        assert result is None

    def test_post_called_with_correct_headers(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("model_call.requests.post", return_value=mock_resp) as mock_post:
            ModelCall.vllm_model_call("history", "text", "question", "NLP")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_payload_contains_user_query(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("model_call.requests.post", return_value=mock_resp) as mock_post:
            ModelCall.vllm_model_call("history", "text", "my specific question", "NLP")
        payload = mock_post.call_args[1]["json"]
        user_messages = [m for m in payload["messages"] if m["role"] == "user"]
        assert any("my specific question" in m["content"] for m in user_messages)


# ─── send_post_request ───────────────────────────────────────────────────────

class TestSendPostRequest:

    @pytest.mark.asyncio
    async def test_yields_text_chunk_from_streamed_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        import json
        chunk_line = "data: " + json.dumps({
            "results": [{"generated_text": "Hello world", "moderations": {}}]
        })

        async def fake_aiter_lines():
            yield chunk_line
            yield ""  # empty line – should be skipped

        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as mock_httpx
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        results = []
        async for item in ModelCall.send_post_request("http://test", {}, {}):
            results.append(item)

        assert any(r["type"] == "text" and r["content"] == "Hello world" for r in results)

    @pytest.mark.asyncio
    async def test_yields_moderation_event_for_flagged_words(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        import json
        chunk_line = "data: " + json.dumps({
            "results": [{
                "generated_text": "",
                "moderations": {
                    "hap": [{"word": "badword"}]
                }
            }]
        })

        async def fake_aiter_lines():
            yield chunk_line

        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as mock_httpx
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        results = []
        async for item in ModelCall.send_post_request("http://test", {}, {}):
            results.append(item)

        assert any(r["type"] == "moderation" and "badword" in r["flagged"] for r in results)
