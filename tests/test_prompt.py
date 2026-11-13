"""
Tests for the prompt selection (prompt.py) and the prompts without sources (attached_prompts.py)

Covers:
  ibm_prompt_formating:
    - every model, prompt type and thinking setting picks the same number with or without attachments,
      the copy without sources for attachments and the current prompt otherwise
    - the number chosen by env is followed for attachments as well

  ATTACHED_PROMPTS:
    - there is one for every prompt that has sources
    - none of them asks for file names or sources, and all can still be formatted
    - each one is the current prompt with lines taken out and nothing else changed, and only lines about
      sources are gone, so a change made to a prompt in prompt.py alone fails here

  ibm_cloud:
    - attached reaches the prompt selection, and is off by default
"""

import re
import pytest
from unittest.mock import patch

from attached_prompts import ATTACHED_PROMPTS
from prompt import CallPrompt

LLAMA, MIXTRAL, SCOUT = "Llama-3.3-70B", "Mixtral-8x7B", "Llama-4-Scout-17B"


def current_prompt(number):
    with patch("prompt.prompt_index_NL", number):
        return CallPrompt.ibm_prompt_formating("NLP", False, LLAMA)


def copy_lines(number):
    """the lines of a copy, with the one line that is shortened rather than dropped (in prompt 14) as the original has it"""
    copy = ATTACHED_PROMPTS[number]
    if number == 14:
        copy = copy.replace("answer with one heading and one short paragraph.", "answer with one heading, one short paragraph, and the sources.")
    return copy.split("\n")


class TestSelection:

    @pytest.mark.parametrize("prompt_type, thinking, model, number", [
        ("NLP", False, LLAMA, 14), ("NLP_U", False, LLAMA, 14), ("EXT", False, LLAMA, 11), ("EXT_U", False, LLAMA, 11),
        ("NLP", True, LLAMA, 13), ("NLP_U", True, LLAMA, 13), ("EXT", True, LLAMA, 13), ("EXT_U", True, LLAMA, 13),
        ("NLP", False, MIXTRAL, 5), ("NLP", True, MIXTRAL, 6), ("NLP", False, SCOUT, 7), ("NLP", True, SCOUT, 8),
    ])
    def test_each_setting_picks_the_same_number_with_or_without_attachments(self, prompt_type, thinking, model, number):
        assert CallPrompt.ibm_prompt_formating(prompt_type, thinking, model, True) == ATTACHED_PROMPTS[number]
        assert CallPrompt.ibm_prompt_formating(prompt_type, thinking, model) == current_prompt(number) != ATTACHED_PROMPTS[number]

    def test_the_number_chosen_by_env_is_followed(self):
        with patch("prompt.prompt_index_NL", 1):
            assert CallPrompt.ibm_prompt_formating("NLP", False, LLAMA, True) == ATTACHED_PROMPTS[1]


class TestAttachedPrompts:

    def test_every_prompt_with_sources_has_a_copy(self):
        assert sorted(ATTACHED_PROMPTS) == [1, 5, 6, 7, 8, 11, 13, 14, 15]

    @pytest.mark.parametrize("number", sorted(ATTACHED_PROMPTS))
    def test_the_current_prompt_asks_for_file_names(self, number):
        assert re.search(r"file ?names?", current_prompt(number), re.IGNORECASE)

    @pytest.mark.parametrize("number", sorted(ATTACHED_PROMPTS))
    def test_nothing_asks_for_file_names_or_sources(self, number):
        prompt = ATTACHED_PROMPTS[number].lower()
        for asked in ("filename", "<font", "source:", "sources:", "list any sources", "copy each file name"):
            assert asked not in prompt

    @pytest.mark.parametrize("number", sorted(ATTACHED_PROMPTS))
    def test_it_still_formats(self, number):
        filled = ATTACHED_PROMPTS[number].format(text="THE_TEXT", query="THE_QUERY", history="THE_HISTORY")
        assert "THE_TEXT" in filled and "THE_QUERY" in filled and "THE_HISTORY" in filled

    @pytest.mark.parametrize("number", sorted(ATTACHED_PROMPTS))
    def test_it_is_the_current_prompt_with_lines_taken_out(self, number):
        lines = iter(current_prompt(number).split("\n"))
        for line in copy_lines(number):
            # `in` moves through the iterator, so the lines also have to be in the same order
            assert line in lines, f"prompt {number} changed: {line!r}"

    @pytest.mark.parametrize("number", sorted(ATTACHED_PROMPTS))
    def test_only_lines_about_sources_are_gone(self, number):
        copy = copy_lines(number)
        removed = [line for line in current_prompt(number).split("\n") if line not in copy]
        assert removed
        for line in removed:
            assert not line.strip() or re.search(r"file ?names?|\bfiles\b|sources?\b", line, re.IGNORECASE), f"prompt {number} lost: {line!r}"


class TestIbmCloud:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs, attached", [({}, False), ({"attached": True}, True)])
    async def test_attached_reaches_the_prompt_selection(self, kwargs, attached):
        from model_call import ModelCall

        async def no_chunks(*args, **_):
            return
            yield

        with patch("model_call.format_prompt") as mock_prompt, patch.object(ModelCall, "send_post_request", no_chunks), \
             patch.object(ModelCall, "get_token", return_value="Bearer t"):
            mock_prompt.ibm_prompt_formating.return_value = "{history}{text}{query}"
            async for _ in ModelCall().ibm_cloud("h", "t", "q", "NLP", LLAMA, False, **kwargs):
                pass
        mock_prompt.ibm_prompt_formating.assert_called_once_with("NLP", False, LLAMA, attached)
