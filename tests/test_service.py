"""
Tests for service.py (UniversalDocutalk)

Covers:
  above_and_below_page_data (static method):
    - middle page: fetches page above + current + page below
    - first page: fetches current + two pages below
    - gracefully handles None responses from Elasticsearch

  search_documents_gpt (async generator):
    - unsupported answer_type yields error message
    - singleDocument with no ES hits yields "not found" message
    - multiDocument with no ES hits yields "not found" message
    - multiDocument with hits but nothing relevant after reranking yields "not found"
    - the answer is streamed first, then one meta_data frame (same format) with only the files the answer cites
    - an answer citing no file, or the fallback message, sends no meta_data
    - nothing passed the reranker and the answer says the documents do not cover it: no meta_data
    - the meta_data sent is logged, and so is an answer for which none is sent
    - a cited fallback chunk (nothing passed the reranker) is sent as meta_data
    - multiDocument references are one per fId, so two files with the same name are both shown
    - a turn with attachments reads them, then searches only them; a turn without never touches them
    - singleDocument on an attachment takes its pages from the attachment index, with the file's other kept chunks
    - nothing readable is said as such, part readable is said before the answer and kept out of the history
    - a greeting or an unsupported answer type reads nothing
    - an answer from attachments uses the prompts without sources and sends no meta_data; every other answer is as before
"""

import json
import logging
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from attachments import AttachedDocument


# ─── above_and_below_page_data ────────────────────────────────────────────────

class TestAboveAndBelowPageData:

    @pytest.fixture(autouse=True)
    def patch_service_es(self):
        with patch("service.es") as mock_es:
            self.mock_es = mock_es
            yield

    def _page(self, page_num):
        return {"merged_text": f"page{page_num}_content"}

    def test_middle_page_combines_above_current_below(self):
        self.mock_es.data_by_page_no.side_effect = lambda p, fid, index: self._page(p)
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("main_text", 5, "file_001")
        assert "page4_content" in result
        assert "main_text" in result
        assert "page6_content" in result

    def test_first_page_combines_current_plus_two_below(self):
        self.mock_es.data_by_page_no.side_effect = lambda p, fid, index: self._page(p)
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("first_page_text", 1, "file_001")
        assert "first_page_text" in result
        assert "page2_content" in result
        assert "page3_content" in result

    def test_middle_page_handles_none_above_page(self):
        def side_effect(page, fid, index):
            return None if page == 4 else self._page(page)
        self.mock_es.data_by_page_no.side_effect = side_effect
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("main_text", 5, "file_001")
        assert "main_text" in result
        assert "page6_content" in result

    def test_middle_page_handles_none_below_page(self):
        def side_effect(page, fid, index):
            return None if page == 6 else self._page(page)
        self.mock_es.data_by_page_no.side_effect = side_effect
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("main_text", 5, "file_001")
        assert "page4_content" in result
        assert "main_text" in result

    def test_first_page_handles_all_none_returns_only_text(self):
        self.mock_es.data_by_page_no.return_value = None
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("only_text", 1, "file_001")
        assert result == "only_text"

    def test_middle_page_handles_all_none_returns_only_text(self):
        self.mock_es.data_by_page_no.return_value = None
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("only_text", 3, "file_001")
        assert result == "only_text"

    def test_page_two_is_a_middle_page_not_first(self):
        self.mock_es.data_by_page_no.side_effect = lambda p, fid, index: self._page(p)
        from service import UniversalDocutalk
        result = UniversalDocutalk.above_and_below_page_data("page2_text", 2, "file_001")
        # page_no != 1 branch: above=page1, below=page3
        assert "page1_content" in result
        assert "page2_text" in result
        assert "page3_content" in result

    def test_drive_is_the_default_index(self):
        self.mock_es.data_by_page_no.return_value = None
        from service import UniversalDocutalk
        import es_call
        UniversalDocutalk.above_and_below_page_data("text", 2, "file_001")
        assert {call.args[2] for call in self.mock_es.data_by_page_no.call_args_list} == {es_call.DOCUTALK_INDEX}


# ─── search_documents_gpt (async generator) ───────────────────────────────────

class TestSearchDocumentsGpt:

    @pytest.fixture(autouse=True)
    def patch_dependencies(self):
        with patch("service.es") as mock_es, \
             patch("service.model") as mock_model, \
             patch("service.ibm_model_query") as mock_enhance:
            self.mock_es = mock_es
            self.mock_model = mock_model
            self.mock_enhance = mock_enhance
            yield

    async def _collect(self, async_gen):
        chunks = []
        async for chunk in async_gen:
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_unsupported_answer_type_yields_error(self):
        from service import UniversalDocutalk
        docutalk = UniversalDocutalk()
        chunks = await self._collect(
            docutalk.search_documents_gpt(
                "query", "user1", "NLP", "INVALID_TYPE",
                "T_path", "q1", "conv1", "Llama-3.3-70B", False, "username"
            )
        )
        assert len(chunks) >= 1
        assert "Unsupported answer type" in chunks[0][0]["response_text"]

    @pytest.mark.asyncio
    async def test_single_document_no_hits_yields_not_found(self):
        self.mock_es.get_conversation_by_id.return_value = None
        self.mock_enhance.return_value = "enhanced query"
        self.mock_es.search_docs_gpt.return_value = []
        from service import UniversalDocutalk
        docutalk = UniversalDocutalk()
        chunks = await self._collect(
            docutalk.search_documents_gpt(
                "query", "user1", "NLP", "singleDocument",
                "T_path", "q1", "conv1", "Llama-3.3-70B", False, "username"
            )
        )
        assert any("couldn't find" in c[0].get("response_text", "") for c in chunks)

    @pytest.mark.asyncio
    async def test_multi_document_no_hits_yields_not_found(self):
        self.mock_es.get_conversation_by_id.return_value = None
        self.mock_enhance.return_value = "['sub1', 'sub2']"
        self.mock_es.search_docs_gpt_split_query.return_value = []
        from service import UniversalDocutalk
        docutalk = UniversalDocutalk()
        chunks = await self._collect(
            docutalk.search_documents_gpt(
                "Compare A and B", "user1", "NLP", "multiDocument",
                "T_path", "q1", "conv1", "Llama-3.3-70B", False, "username"
            )
        )
        assert any("couldn't find" in c[0].get("response_text", "") for c in chunks)

    @pytest.mark.asyncio
    async def test_multi_document_no_relevant_hits_after_rerank_yields_not_found(self):
        self.mock_es.get_conversation_by_id.return_value = None
        self.mock_enhance.return_value = "['sub1']"
        self.mock_es.search_docs_gpt_split_query.return_value = [
            {
                "_score": 0.01,
                "_id": "doc1",
                "_source": {
                    "fileName": "file.pdf",
                    "fId": "fid1",
                    "text": "some text",
                    "pageNo": "1"
                }
            }
        ]
        from service import UniversalDocutalk
        docutalk = UniversalDocutalk()
        # reranker finds nothing relevant among the hits
        with patch("service.rerank_hits", return_value=([], False)) as mock_rerank:
            chunks = await self._collect(
                docutalk.search_documents_gpt(
                    "query", "user1", "NLP", "multiDocument",
                    "T_path", "q1", "conv1", "Llama-3.3-70B", False, "username"
                )
            )
        # reranked against the rewritten question, not the raw query
        assert mock_rerank.call_args[0][0] == "sub1"
        assert any("find information on this topic" in c[0].get("response_text", "") for c in chunks)

    @staticmethod
    def _sources(*names):
        """the Sources line the prompts ask for"""
        return "<p><strong>Sources:</strong> " + ", ".join(f'<font color="#6366F1">{name}</font>' for name in names) + "</p>"

    async def _stream_frames(self, answer_type, reranked, answer_chunks=("Hello", " world")):
        """runs the service with rerank_hits returning `reranked` and the model streaming `answer_chunks`;
        frames are serialised as soon as they are yielded, like main.py does"""
        self.mock_es.get_conversation_by_id.return_value = None
        self.mock_es.data_by_page_no.return_value = None
        self.mock_enhance.return_value = "['sub1']"
        self.mock_es.search_docs_gpt.return_value = reranked[0]
        self.mock_es.search_docs_gpt_split_query.return_value = reranked[0]

        async def fake_stream(*args, **kwargs):
            for chunk in answer_chunks:
                yield chunk
        self.mock_model.ibm_cloud = fake_stream

        from service import UniversalDocutalk
        frames = []
        with patch("service.rerank_hits", return_value=reranked):
            async for frame in UniversalDocutalk().search_documents_gpt(
                "query", "user1", "NLP", answer_type,
                "T_path", "q1", "conv1", "Llama-3.3-70B", False, "username"
            ):
                frames.append(json.loads(json.dumps(frame)))
        return frames

    @staticmethod
    def _hit(hit_id, fid, filename, page_no="1", score=0.9):
        return {"_score": score, "_id": hit_id, "_source": {"fileName": filename, "fId": fid, "text": f"text of {hit_id}", "pageNo": page_no}}

    @pytest.mark.asyncio
    async def test_multi_document_streams_answer_then_meta_data_of_cited_files(self):
        sources = self._sources("file.pdf")
        frames = await self._stream_frames("multiDocument", ([self._hit("doc1", "fid1", "file.pdf", score=0.91)], True),
                                           ("Hello", sources))
        # the answer first, then one meta_data frame in the same format as before
        assert frames == [
            [{"response_text": "Hello"}],
            [{"response_text": sources}],
            [{"meta_data": [{"filename": "file.pdf", "username": "user1", "fId": "fid1", "page_no": "1", "score": 0.91,
                             "id": "doc1", "query_id": "q1", "query": "query", "status": 0, "text": "text of doc1"}]}],
        ]
        self.mock_es.store_chat_history.assert_called_once_with("conv1", "query", "Hello" + sources)

    @pytest.mark.asyncio
    async def test_multi_document_sends_only_the_cited_files(self):
        hits = [self._hit("a_1_0", "a", "a.pdf"), self._hit("b_1_0", "b", "b.pdf"), self._hit("c_1_0", "c", "c.pdf")]
        frames = await self._stream_frames("multiDocument", (hits, True), ("Answer from b and c. ", self._sources("c.pdf", "b.pdf")))
        assert [r["fId"] for r in frames[-1][0]["meta_data"]] == ["c", "b"]

    @pytest.mark.asyncio
    async def test_multi_document_answer_citing_no_file_sends_no_meta_data(self):
        frames = await self._stream_frames("multiDocument", ([self._hit("a_1_0", "a", "a.pdf")], True))
        assert frames == [[{"response_text": "Hello"}], [{"response_text": " world"}]]

    @pytest.mark.asyncio
    async def test_multi_document_fallback_answer_sends_no_meta_data(self):
        fallback = "<p><em>Thank you for your search! I couldn't find information on this topic in your uploaded documents.</em></p>"
        frames = await self._stream_frames("multiDocument", ([self._hit("a_1_0", "a", "a.pdf")], False),
                                           (fallback, self._sources("a.pdf")))
        assert all("meta_data" not in frame[0] for frame in frames)

    @pytest.mark.asyncio
    async def test_multi_document_references_are_one_per_file_id(self):
        hits = [self._hit("a_1_0", "fid-a", "report.pdf"), self._hit("a_2_0", "fid-a", "report.pdf", "2"),
                self._hit("b_1_0", "fid-b", "report.pdf"), self._hit("c_1_0", "fid-c", "other.pdf")]
        frames = await self._stream_frames("multiDocument", (hits, True), ("Answer. ", self._sources("report.pdf", "other.pdf")))
        # two different files called report.pdf are both shown; the second chunk of fid-a adds no reference
        assert [(r["fId"], r["filename"], r["id"]) for r in frames[-1][0]["meta_data"]] == [
            ("fid-a", "report.pdf", "a_1_0"), ("fid-b", "report.pdf", "b_1_0"), ("fid-c", "other.pdf", "c_1_0")]

    @pytest.mark.asyncio
    async def test_single_document_streams_answer_then_meta_data_of_cited_file(self):
        frames = await self._stream_frames("singleDocument", ([self._hit("a_3_0", "a", "a.pdf", "3")], True),
                                           ("Hello", self._sources("a.pdf")))
        assert frames[0] == [{"response_text": "Hello"}]
        assert [r["fId"] for r in frames[-1][0]["meta_data"]] == ["a"]

    @pytest.mark.asyncio
    async def test_single_document_cited_fallback_chunk_is_sent_as_meta_data(self):
        # nothing passed the reranker, but the model used the fallback chunk and cites it
        frames = await self._stream_frames("singleDocument", ([self._hit("a_3_0", "a", "a.pdf", "3")], False),
                                           ("Hello", self._sources("a.pdf")))
        assert [r["fId"] for r in frames[-1][0]["meta_data"]] == ["a"]

    @pytest.mark.asyncio
    async def test_single_document_answer_citing_no_file_sends_no_meta_data(self):
        frames = await self._stream_frames("singleDocument", ([self._hit("a_3_0", "a", "a.pdf", "3")], True))
        assert frames == [[{"response_text": "Hello"}], [{"response_text": " world"}]]

    @pytest.mark.asyncio
    async def test_fallback_chunks_and_not_covered_answer_send_no_meta_data(self):
        # nothing passed the reranker, the model says the documents do not cover it but still lists the files
        frames = await self._stream_frames("multiDocument", ([self._hit("a_1_0", "a", "a.pdf")], False),
                                           ("<p>The reference text does not contain a recipe.</p>", self._sources("a.pdf")))
        assert all("meta_data" not in frame[0] for frame in frames)

    @pytest.mark.asyncio
    async def test_meta_data_sent_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="service"):
            await self._stream_frames("multiDocument", ([self._hit("a_1_0", "a", "a.pdf")], True), ("Hello", self._sources("a.pdf")))
        sent = [message for message in caplog.messages if message.startswith("meta_data sent --> ")]
        assert len(sent) == 1 and "'filename': 'a.pdf'" in sent[0]

    @pytest.mark.asyncio
    async def test_meta_data_not_sent_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="service"):
            await self._stream_frames("singleDocument", ([self._hit("a_3_0", "a", "a.pdf", "3")], True))
        assert "meta_data not sent: no reference file for this answer" in caplog.messages


# ─── attached documents ───────────────────────────────────────────────────────

class TestAttachments:

    ATTACHED = [AttachedDocument(fid="abc", miniourl="dms//abc"), AttachedDocument(fid="def", miniourl="dms//def")]

    @pytest.fixture(autouse=True)
    def patch_dependencies(self):
        with patch("service.es") as mock_es, \
             patch("service.model") as mock_model, \
             patch("service.ibm_model_query", return_value='["rewritten"]'), \
             patch("service.ingest_attachments", return_value=(["abc", "def"], [])) as mock_ingest, \
             patch("service.rerank_hits", side_effect=lambda query, hits: (hits, True)):
            self.mock_es = mock_es
            self.mock_ingest = mock_ingest
            self.contexts = []
            self.attached = []
            self.answer = "Answer"

            async def fake_stream(history, text, *args):
                self.contexts.append(text)
                self.attached.append(args[-1])
                yield self.answer
            mock_model.ibm_cloud = fake_stream
            mock_es.get_conversation_by_id.return_value = None
            mock_es.data_by_page_no.return_value = None
            mock_es.search_attachments.return_value = [self._hit("c1", "abc", "best chunk", "2"), self._hit("c2", "abc", "second chunk", "9"),
                                                       self._hit("c3", "def", "other file chunk", "1")]
            yield

    @staticmethod
    def _hit(hit_id, fid, text, page_no):
        return {"_score": 0.9, "_id": hit_id, "_source": {"fileName": f"{fid}.pdf", "fId": fid, "text": text, "pageNo": page_no}}

    async def _frames(self, answer_type="multiDocument", attachments=ATTACHED, query="what is this about"):
        from service import UniversalDocutalk
        generator = UniversalDocutalk().search_documents_gpt(query, "user1", "NLP", answer_type, "T_path", "q1", "conv1",
                                                             "Llama-3.3-70B", False, "username", attachments)
        return [frame async for frame in generator]

    @pytest.mark.asyncio
    async def test_multi_document_reads_the_files_then_searches_only_them(self):
        await self._frames()
        self.mock_ingest.assert_called_once_with(self.ATTACHED, "user1")
        self.mock_es.search_attachments.assert_called_once_with(["rewritten", "what is this about"], ["abc", "def"])
        self.mock_es.search_docs_gpt_split_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_document_reads_the_files_then_searches_only_them(self):
        await self._frames("singleDocument")
        self.mock_es.search_attachments.assert_called_once_with(['["rewritten"]'], ["abc", "def"])
        self.mock_es.search_docs_gpt.assert_not_called()

    @pytest.mark.asyncio
    async def test_scope_does_not_limit_attachments(self):
        # scope "none" returns [] from the drive search; the fIds decide what is searched here
        from service import UniversalDocutalk
        generator = UniversalDocutalk().search_documents_gpt("q?", "user1", "NLP", "multiDocument", "T_path", "q1", "conv1",
                                                             "Llama-3.3-70B", False, "none", self.ATTACHED)
        assert [frame async for frame in generator][0] == [{"response_text": "Answer"}]

    @pytest.mark.asyncio
    async def test_single_document_pages_come_from_the_attachment_index(self):
        import es_call
        await self._frames("singleDocument")
        assert {call.args[2] for call in self.mock_es.data_by_page_no.call_args_list} == {es_call.ATTACHMENT_INDEX}

    @pytest.mark.asyncio
    async def test_single_document_gets_the_files_other_kept_chunks(self):
        # all of a docx is page 1, so with no page around it the best chunk would otherwise be the whole context
        await self._frames("singleDocument")
        assert "best chunk second chunk" in self.contexts[0]
        assert "other file chunk" not in self.contexts[0]

    @pytest.mark.asyncio
    async def test_a_turn_without_attachments_searches_the_drive(self):
        self.mock_es.search_docs_gpt_split_query.return_value = []
        for attachments in (None, []):
            await self._frames(attachments=attachments)
        self.mock_ingest.assert_not_called()
        self.mock_es.search_attachments.assert_not_called()
        assert self.mock_es.search_docs_gpt_split_query.call_count == 2

    @pytest.mark.asyncio
    async def test_nothing_readable_says_so_and_stops(self):
        from service import ATTACHMENTS_UNREADABLE
        self.mock_ingest.return_value = ([], ["abc", "def"])
        assert await self._frames() == [[{"response_text": ATTACHMENTS_UNREADABLE}]]
        self.mock_es.search_attachments.assert_not_called()
        self.mock_es.store_chat_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_part_readable_says_so_first_and_searches_the_rest(self):
        self.mock_ingest.return_value = (["abc"], ["def"])
        frames = await self._frames()
        assert "1 of the 2 attached documents" in frames[0][0]["response_text"]
        assert frames[1] == [{"response_text": "Answer"}]
        assert self.mock_es.search_attachments.call_args.args[1] == ["abc"]
        # the note is about this turn, the history keeps the answer
        self.mock_es.store_chat_history.assert_called_once_with("conv1", "what is this about", "Answer")

    @pytest.mark.asyncio
    async def test_no_hits_in_the_attachments_is_the_not_found_message(self):
        self.mock_es.search_attachments.return_value = []
        frames = await self._frames()
        assert "find information on this topic" in frames[0][0]["response_text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("answer_type", ["singleDocument", "multiDocument"])
    @pytest.mark.parametrize("history", [None, "Question: a\nAnswer: b"])
    async def test_attached_documents_use_the_prompts_without_sources(self, answer_type, history):
        self.mock_es.get_conversation_by_id.return_value = history
        await self._frames(answer_type)
        assert self.attached == [True]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("answer_type", ["singleDocument", "multiDocument"])
    @pytest.mark.parametrize("history", [None, "Question: a\nAnswer: b"])
    async def test_without_attachments_the_current_prompts_are_used(self, answer_type, history):
        self.mock_es.get_conversation_by_id.return_value = history
        drive_hit = [self._hit("d1", "drive", "drive chunk", "1")]
        self.mock_es.search_docs_gpt.return_value = drive_hit
        self.mock_es.search_docs_gpt_split_query.return_value = drive_hit
        await self._frames(answer_type, attachments=None)
        assert self.attached == [False]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("answer_type", ["singleDocument", "multiDocument"])
    async def test_no_meta_data_even_if_the_model_cites_an_attached_file(self, answer_type):
        self.answer = '<p>Answer.</p><p><strong>Sources:</strong> <font color="#6366F1">abc.pdf</font></p>'
        frames = await self._frames(answer_type)
        assert all("meta_data" not in frame[0] for frame in frames)

    @pytest.mark.asyncio
    async def test_without_attachments_the_meta_data_is_still_sent(self):
        self.answer = '<p>Answer.</p><p><strong>Sources:</strong> <font color="#6366F1">drive.pdf</font></p>'
        self.mock_es.search_docs_gpt_split_query.return_value = [self._hit("d1", "drive", "drive chunk", "1")]
        frames = await self._frames(attachments=None)
        assert frames[-1][0]["meta_data"][0]["fId"] == "drive"

    @pytest.mark.asyncio
    async def test_a_greeting_reads_nothing(self):
        await self._frames(query="hello")
        self.mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unsupported_answer_type_reads_nothing(self):
        await self._frames("INVALID_TYPE")
        self.mock_ingest.assert_not_called()
