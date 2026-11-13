"""
Tests for attachments.py

Covers:
  AttachedDocument:
    - only fid is needed, every value is trimmed, a blank fid is refused

  attached_documents:
    - the documents of a bare array or of {"documents": [...]}, in the order given
    - no body, no documents, or no miniourl on any document is no attachment
    - a miniourl on some documents only raises
    - a repeated fid keeps its first url, too many files raise

  ingest_attachments:
    - every file goes to the attachment index with the username and its fileName, and no tenant is asked for
    - indexed, partial and already_indexed can be searched; empty and failed cannot
    - an unusable index reads nothing and reports every file as not read

  doc_ingest.ingest_one (the file name):
    - the name minio recorded wins, then the fileName sent in, then the object's own name
    - the download and the indexing are logged with the fId, and neither for a file already in the index
"""

import logging
import pytest
from unittest.mock import patch
from pydantic import ValidationError

import attachments
import doc_ingest
from attachments import AttachedDocument, AttachmentBody, attached_documents, ingest_attachments
from es_call import ATTACHMENT_INDEX


def document(fid, miniourl=None, fileName=""):
    return AttachedDocument(fid=fid, miniourl=f"dms//{fid}" if miniourl is None else miniourl, fileName=fileName)


class TestAttachedDocument:

    def test_only_fid_is_needed(self):
        assert AttachedDocument(fid="a").model_dump() == {"fid": "a", "miniourl": "", "fileName": ""}

    def test_values_are_trimmed(self):
        attached = AttachedDocument(fid=" a ", miniourl=" dms//a ", fileName=" a.pdf ")
        assert attached.model_dump() == {"fid": "a", "miniourl": "dms//a", "fileName": "a.pdf"}

    @pytest.mark.parametrize("fid", ["", "   "])
    def test_a_blank_fid_is_refused(self, fid):
        with pytest.raises(ValidationError):
            AttachedDocument(fid=fid)


class TestAttachedDocuments:

    def test_bare_array_in_the_order_given(self):
        documents = [document("b"), document("a")]
        assert attached_documents(documents) == documents

    def test_object_form(self):
        documents = [document("a")]
        assert attached_documents(AttachmentBody(documents=documents)) == documents

    @pytest.mark.parametrize("body", [None, [], AttachmentBody(), AttachmentBody(documents=[])])
    def test_nothing_sent_is_no_attachment(self, body):
        assert attached_documents(body) == []

    def test_no_url_on_any_document_is_no_attachment(self, caplog):
        with caplog.at_level(logging.WARNING, logger="attachments"):
            assert attached_documents([document("a", ""), document("b", "")]) == []
        assert "nothing is attached" in caplog.text

    def test_a_url_on_some_documents_only_raises(self):
        with pytest.raises(ValueError, match="'b' has none"):
            attached_documents([document("a"), document("b", "")])

    def test_a_repeated_fid_keeps_its_first_url(self):
        first = document("a", "dms//first")
        assert attached_documents([first, document("a", "dms//second")]) == [first]

    def test_too_many_raise(self):
        with pytest.raises(ValueError, match=f"at most {attachments.MAX_ATTACHMENTS}"):
            attached_documents([document(str(i)) for i in range(attachments.MAX_ATTACHMENTS + 1)])

    def test_the_limit_itself_is_accepted(self):
        assert len(attached_documents([document(str(i)) for i in range(attachments.MAX_ATTACHMENTS)])) == attachments.MAX_ATTACHMENTS


class TestIngestAttachments:

    ATTACHED = [document("a", fileName="a.pdf"), document("b")]

    @pytest.fixture(autouse=True)
    def mock_ingest(self):
        with patch("attachments.doc_ingest") as mock_ingest:
            self.mock_ingest = mock_ingest
            yield

    def _statuses(self, *statuses):
        self.mock_ingest.ingest_one.side_effect = [{"status": status, "chunks": 0, "reason": None, "fileName": "a.pdf"} for status in statuses]

    def test_every_file_goes_to_the_attachment_index_with_the_username_and_its_name(self):
        self._statuses("indexed", "indexed")
        ingest_attachments(self.ATTACHED, "user1")
        self.mock_ingest.ensure_index.assert_called_once_with(ATTACHMENT_INDEX)
        calls = self.mock_ingest.ingest_one.call_args_list
        assert [call.args for call in calls] == [("a", "dms//a"), ("b", "dms//b")]
        assert [call.kwargs for call in calls] == [{"index": ATTACHMENT_INDEX, "username": "user1", "file_name": "a.pdf"},
                                                   {"index": ATTACHMENT_INDEX, "username": "user1", "file_name": ""}]

    @pytest.mark.parametrize("status", ["indexed", "partial", "already_indexed"])
    def test_a_file_with_chunks_in_the_index_is_readable(self, status):
        self._statuses(status)
        assert ingest_attachments(self.ATTACHED[:1], "user1") == (["a"], [])

    @pytest.mark.parametrize("status", ["empty", "failed"])
    def test_a_file_with_no_chunks_is_not_read(self, status):
        self._statuses(status)
        assert ingest_attachments(self.ATTACHED[:1], "user1") == ([], ["a"])

    def test_the_outcome_of_each_file_is_logged(self, caplog):
        self.mock_ingest.ingest_one.side_effect = [{"status": "indexed", "chunks": 11, "reason": None, "fileName": "a.pdf"}]
        with caplog.at_level(logging.INFO, logger="attachments"):
            ingest_attachments(self.ATTACHED[:1], "user1")
        assert f"Attachment a ('a.pdf') --> indexed, 11 chunk(s) written to '{ATTACHMENT_INDEX}'" in caplog.text

    def test_one_bad_file_does_not_cost_the_others(self):
        self._statuses("failed", "indexed")
        assert ingest_attachments(self.ATTACHED, "user1") == (["b"], ["a"])

    def test_an_unusable_index_reads_nothing(self, caplog):
        self.mock_ingest.ensure_index.side_effect = RuntimeError("no such pipeline")
        with caplog.at_level(logging.ERROR, logger="attachments"):
            assert ingest_attachments(self.ATTACHED, "user1") == ([], ["a", "b"])
        self.mock_ingest.ingest_one.assert_not_called()
        assert "no such pipeline" in caplog.text


class TestFileName:

    def _ingest(self, stored_name, sent_name):
        """the name the file is read and indexed under"""
        with patch("doc_ingest.indexed_fids", return_value=set()), \
             patch("doc_ingest.download", return_value=(b"data", stored_name)), \
             patch("doc_ingest.extract_chunks", return_value=[{"pageNo": 1, "para": 0, "text": "text"}]) as extract, \
             patch("doc_ingest.bulk_index", return_value=(1, [])) as bulk:
            report = doc_ingest.ingest_one("6a96", "dms//6a96", index="i", file_name=sent_name)
        assert report["status"] == "indexed"
        name = extract.call_args.kwargs["file_name"]
        assert bulk.call_args.args[0][0]["fileName"] == report["fileName"] == name
        return name

    def test_the_name_minio_recorded_wins(self):
        assert self._ingest("recorded.pdf", "sent.pdf") == "recorded.pdf"

    def test_the_name_sent_in_beats_the_object_name(self):
        assert self._ingest("6a96", "sent.pdf") == "sent.pdf"

    def test_the_object_name_is_the_last_resort(self):
        assert self._ingest("6a96", "") == "6a96"


class TestIngestLogs:

    def test_download_and_indexing_are_logged_in_order_with_the_fid(self, caplog):
        with patch("doc_ingest.indexed_fids", return_value=set()), \
             patch("doc_ingest.download", return_value=(b"data", "6a96")), \
             patch("doc_ingest.extract_chunks", return_value=[{"pageNo": 1, "para": 0, "text": "text"}]), \
             patch("doc_ingest.bulk_index", return_value=(1, [])), \
             caplog.at_level(logging.INFO, logger="doc_ingest"):
            doc_ingest.ingest_one("6a96", "dms//6a96", index="docutalk")
        assert [message for message in caplog.messages if "fId=6a96" in message][:2] == [
            "Downloading fId=6a96 from minio : dms//6a96", "Indexing 1 chunk(s) of fId=6a96 into 'docutalk'"]

    def test_a_file_already_in_the_index_never_says_downloading(self, caplog):
        with patch("doc_ingest.indexed_fids", return_value={"6a96"}), \
             caplog.at_level(logging.INFO, logger="doc_ingest"):
            doc_ingest.ingest_one("6a96", "dms//6a96", index="docutalk")
        assert not [message for message in caplog.messages if message.startswith(("Downloading", "Indexing"))]
        assert "fId=6a96 is already indexed in 'docutalk', skipping" in caplog.messages
