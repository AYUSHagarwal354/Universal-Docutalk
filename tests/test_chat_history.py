"""
Tests for chat_history.py (RedisConnections)

Covers:
  - update_memory: stores correct JSON in Redis, trims to MEMORY_SIZE
  - get_chat_history_prompt:
      with query_enhancement=True: includes question only
      with query_enhancement=False: includes question and answer
      no history: is_history stays False
  - reset_memory: calls redis delete with correct key
"""

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def redis_conn():
    """Return a RedisConnections instance with the Redis client mocked."""
    with patch("chat_history.redis.Redis") as mock_redis_class:
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        from chat_history import RedisConnections
        conn = RedisConnections()
        conn.redis_client = mock_client
        return conn


# ─── update_memory ────────────────────────────────────────────────────────────

class TestUpdateMemory:

    def test_pushes_serialised_qa_pair_to_redis(self, redis_conn):
        redis_conn.update_memory("session_key_1", "What is Redis?", "An in-memory DB")
        redis_conn.redis_client.rpush.assert_called_once()
        key, raw = redis_conn.redis_client.rpush.call_args[0]
        assert key == "session_key_1"
        stored = json.loads(raw)
        assert stored["question"] == "What is Redis?"
        assert stored["answer"] == "An in-memory DB"

    def test_trims_list_to_memory_size_of_five(self, redis_conn):
        redis_conn.update_memory("key1", "q", "a")
        redis_conn.redis_client.ltrim.assert_called_once_with("key1", -5, -1)

    def test_called_for_multiple_sessions_independently(self, redis_conn):
        redis_conn.update_memory("session_A", "q1", "a1")
        redis_conn.update_memory("session_B", "q2", "a2")
        assert redis_conn.redis_client.rpush.call_count == 2
        keys = [c[0][0] for c in redis_conn.redis_client.rpush.call_args_list]
        assert "session_A" in keys
        assert "session_B" in keys


# ─── get_chat_history_prompt ─────────────────────────────────────────────────

class TestGetChatHistoryPrompt:

    def _make_qa(self, question, answer):
        return json.dumps({"question": question, "answer": answer})

    def test_query_enhancement_true_includes_question_only(self, redis_conn):
        redis_conn.redis_client.lrange.return_value = [
            self._make_qa("What is Elasticsearch?", "A distributed search engine")
        ]
        prompt, is_history = redis_conn.get_chat_history_prompt("session_1", True)
        assert is_history is True
        assert "What is Elasticsearch?" in prompt
        # answer should NOT appear when query_enhancement=True
        assert "A distributed search engine" not in prompt

    def test_query_enhancement_false_includes_question_and_answer(self, redis_conn):
        redis_conn.redis_client.lrange.return_value = [
            self._make_qa("What is Redis?", "An in-memory data store")
        ]
        prompt, is_history = redis_conn.get_chat_history_prompt("session_1", False)
        assert is_history is True
        assert "What is Redis?" in prompt
        assert "An in-memory data store" in prompt

    def test_empty_redis_returns_false_is_history(self, redis_conn):
        redis_conn.redis_client.lrange.return_value = []
        prompt, is_history = redis_conn.get_chat_history_prompt("session_1", True)
        assert is_history is False

    def test_multiple_items_all_included_in_prompt(self, redis_conn):
        redis_conn.redis_client.lrange.return_value = [
            self._make_qa("Q1", "A1"),
            self._make_qa("Q2", "A2"),
        ]
        prompt, is_history = redis_conn.get_chat_history_prompt("session_1", False)
        assert "Q1" in prompt
        assert "Q2" in prompt
        assert is_history is True

    def test_lrange_called_with_correct_key(self, redis_conn):
        redis_conn.redis_client.lrange.return_value = []
        redis_conn.get_chat_history_prompt("my_session_key", True)
        redis_conn.redis_client.lrange.assert_called_once_with("my_session_key", -1, -1)


# ─── reset_memory ─────────────────────────────────────────────────────────────

class TestResetMemory:

    def test_calls_redis_delete_with_correct_key(self, redis_conn):
        redis_conn.reset_memory("session_key_abc")
        redis_conn.redis_client.delete.assert_called_once_with("session_key_abc")

    def test_reset_does_not_raise_on_nonexistent_key(self, redis_conn):
        redis_conn.redis_client.delete.return_value = 0  # Redis returns 0 if key did not exist
        redis_conn.reset_memory("nonexistent_key")
        redis_conn.redis_client.delete.assert_called_once()
