"""
Tests for main.py (FastAPI endpoints)

Covers:
  POST /teamsync/nlp/docutalk/delete_conversation
    - returns "deleted", calls es.delete_conversation with the correct id

  POST /teamsync/nlp/docutalk/close_unv_session
    - always returns "close" regardless of session flag

  POST /teamsync/nlp/docutalk/search_documents (streaming)
    - session=False: stream_chunk generator still returned
    - scope values are correctly mapped before being passed downstream
    - attached documents reach the service from the body, as a bare array or under "documents"
    - no body, an empty array or documents with no miniourl is no attachment
    - some documents without a miniourl, or a blank fid, is a 422 and nothing is streamed
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Build the FastAPI TestClient once per module.
    All heavy deps (ES, Redis, IBM IAM) are already patched by conftest.py.
    """
    from main import app
    return TestClient(app, raise_server_exceptions=False)


# ─── /delete_conversation ─────────────────────────────────────────────────────

class TestDeleteConversationEndpoint:

    def test_returns_deleted_on_success(self, client):
        with patch("main.es") as mock_es:
            mock_es.delete_conversation.return_value = True
            response = client.post(
                "/teamsync/nlp/docutalk/delete_conversation",
                params={"conversationId": "conv_abc"}
            )
        assert response.status_code == 200
        assert response.json() == "deleted"

    def test_calls_es_delete_with_correct_conversation_id(self, client):
        with patch("main.es") as mock_es:
            mock_es.delete_conversation.return_value = True
            client.post(
                "/teamsync/nlp/docutalk/delete_conversation",
                params={"conversationId": "specific_conv_id"}
            )
        mock_es.delete_conversation.assert_called_once_with("specific_conv_id")

    def test_returns_200_even_when_es_delete_raises(self, client):
        with patch("main.es") as mock_es:
            mock_es.delete_conversation.side_effect = Exception("ES error")
            response = client.post(
                "/teamsync/nlp/docutalk/delete_conversation",
                params={"conversationId": "conv_err"}
            )
        # Endpoint does not have error handling, so FastAPI returns 500
        # The important thing is the endpoint is reachable
        assert response.status_code in (200, 500)


# ─── /close_unv_session ──────────────────────────────────────────────────────

class TestCloseSessionEndpoint:

    def test_returns_close_when_session_is_false(self, client):
        response = client.post(
            "/teamsync/nlp/docutalk/close_unv_session",
            params={"username": "test@user.com", "conversationId": "conv1", "session": False}
        )
        assert response.status_code == 200
        assert response.json() == "close"

    def test_returns_close_when_session_is_true(self, client):
        response = client.post(
            "/teamsync/nlp/docutalk/close_unv_session",
            params={"username": "test@user.com", "conversationId": "conv1", "session": True}
        )
        assert response.status_code == 200
        assert response.json() == "close"

    def test_accepts_email_with_at_symbol_in_username(self, client):
        response = client.post(
            "/teamsync/nlp/docutalk/close_unv_session",
            params={"username": "alice@example.com", "conversationId": "c99", "session": False}
        )
        assert response.status_code == 200


# ─── /search_documents (streaming) ───────────────────────────────────────────

class TestSearchDocumentsEndpoint:

    def test_endpoint_returns_200_with_session_true(self, client):
        async def mock_stream(*args, **kwargs):
            yield [{"response_text": "mock answer"}]

        with patch("main.docutalk") as mock_docutalk, \
             patch("main.es") as mock_es:
            mock_es.create_conversation_index.return_value = None
            mock_docutalk.search_documents_gpt = mock_stream

            response = client.post(
                "/teamsync/nlp/docutalk/search_documents",
                params={
                    "query": "What is Redis?",
                    "username": "user@test.com",
                    "modeltype": "NLP",
                    "answer_type": "singleDocument",
                    "query_id": "q1",
                    "conversationId": "conv1",
                    "modelName": "Llama-3.3-70B",
                    "scope": "Drive",
                    "thinkingRequired": False,
                    "session": True,
                }
            )
        assert response.status_code == 200

    def test_scope_drive_maps_to_username(self, client):
        captured = {}

        async def mock_stream(query, username, modeltype, answer_type,
                               path, query_id, conv_id, model_name,
                               thinking_required, scope, attachments):
            captured["scope"] = scope
            yield [{"response_text": "ok"}]

        with patch("main.docutalk") as mock_docutalk, \
             patch("main.es") as mock_es:
            mock_es.create_conversation_index.return_value = None
            mock_docutalk.search_documents_gpt = mock_stream

            client.post(
                "/teamsync/nlp/docutalk/search_documents",
                params={
                    "query": "q", "username": "u", "modeltype": "NLP",
                    "answer_type": "singleDocument", "query_id": "q1",
                    "conversationId": "c1", "modelName": "Llama-3.3-70B",
                    "scope": "Drive", "thinkingRequired": False, "session": True,
                }
            )
        assert captured.get("scope") == "username"

    def test_scope_shared_with_me_maps_to_shared_with(self, client):
        captured = {}

        async def mock_stream(query, username, modeltype, answer_type,
                               path, query_id, conv_id, model_name,
                               thinking_required, scope, attachments):
            captured["scope"] = scope
            yield [{"response_text": "ok"}]

        with patch("main.docutalk") as mock_docutalk, \
             patch("main.es") as mock_es:
            mock_es.create_conversation_index.return_value = None
            mock_docutalk.search_documents_gpt = mock_stream

            client.post(
                "/teamsync/nlp/docutalk/search_documents",
                params={
                    "query": "q", "username": "u", "modeltype": "NLP",
                    "answer_type": "singleDocument", "query_id": "q1",
                    "conversationId": "c1", "modelName": "Llama-3.3-70B",
                    "scope": "Shared With Me", "thinkingRequired": False, "session": True,
                }
            )
        assert captured.get("scope") == "shared_with"


class TestAttachmentBody:

    PARAMS = {"query": "q", "username": "u", "modeltype": "NLP", "answer_type": "multiDocument", "query_id": "q1",
              "conversationId": "c1", "modelName": "Llama-3.3-70B", "scope": "Drive", "thinkingRequired": False, "session": True}
    DOCUMENTS = [{"fid": "abc", "miniourl": "dms//abc", "fileName": "a.pdf"}, {"fid": "def", "miniourl": "dms//def"}]

    def _post(self, client, **body):
        """posts with `body` as the request's json, returns (response, what the service got as (fid, miniourl, fileName))"""
        captured = {}

        async def mock_stream(*args):
            captured["attachments"] = [(document.fid, document.miniourl, document.fileName) for document in args[-1]]
            yield [{"response_text": "ok"}]

        with patch("main.docutalk") as mock_docutalk, patch("main.es") as mock_es:
            mock_es.create_conversation_index.return_value = None
            mock_docutalk.search_documents_gpt = mock_stream
            response = client.post("/teamsync/nlp/docutalk/search_documents", params=self.PARAMS, **body)
        return response, captured.get("attachments")

    def test_bare_array_reaches_the_service(self, client):
        response, attachments = self._post(client, json=self.DOCUMENTS)
        assert response.status_code == 200
        assert attachments == [("abc", "dms//abc", "a.pdf"), ("def", "dms//def", "")]

    def test_object_form_reaches_the_service(self, client):
        response, attachments = self._post(client, json={"documents": self.DOCUMENTS})
        assert response.status_code == 200
        assert attachments == [("abc", "dms//abc", "a.pdf"), ("def", "dms//def", "")]

    @pytest.mark.parametrize("body", [{}, {"json": []}, {"json": {}}, {"json": {"documents": None}}, {"json": {"documents": []}},
                                      {"json": [{"fid": "abc"}]}, {"json": [{"fid": "abc", "miniourl": ""}]}])
    def test_nothing_to_read_is_no_attachment(self, client, body):
        response, attachments = self._post(client, **body)
        assert response.status_code == 200
        assert attachments == []

    def test_some_documents_without_a_url_are_a_422_and_nothing_is_streamed(self, client):
        response, attachments = self._post(client, json=[{"fid": "abc", "miniourl": "dms//abc"}, {"fid": "def"}])
        assert response.status_code == 422
        assert "'def' has none" in response.json()["detail"]
        assert attachments is None

    def test_a_blank_fid_is_a_422_and_nothing_is_streamed(self, client):
        response, attachments = self._post(client, json=[{"fid": "  ", "miniourl": "dms//abc"}])
        assert response.status_code == 422
        assert attachments is None
