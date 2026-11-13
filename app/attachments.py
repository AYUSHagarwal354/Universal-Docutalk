"""
    documents attached to a chat turn
    - they arrive in the JSON body in the shape the correspondence service takes them: one object per file, carrying
      its fid, miniourl and fileName
    - doc_ingest downloads each file from minio, reads it (OCR for scans), cuts it into chunks and indexes them into
      ATTACHMENT_INDEX, the same way the correspondence service does for doc_mode=local
    - a file already in the index is skipped, so only the first question on a file is slow
"""
import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

import doc_ingest
from es_call import ATTACHMENT_INDEX

logger = logging.getLogger(__name__)

MAX_ATTACHMENTS = int(os.getenv("MAX_ATTACHMENTS", 20))
# "empty" is not here: the file was read and has no text to search
READABLE_STATUSES = ("indexed", "partial", "already_indexed")


class AttachedDocument(BaseModel):
    """one attached file: its id and where it lives, in one object so a url cannot end up on another file's id"""
    fid: str = Field(..., min_length=1)
    miniourl: str = ""
    # only a fallback: the name minio recorded for the upload wins
    fileName: str = ""

    @field_validator("fid", "miniourl", "fileName")
    @classmethod
    def trimmed(cls, value):
        return str(value).strip()

    @field_validator("fid")
    @classmethod
    def fid_not_blank(cls, fid):
        if not fid:
            raise ValueError("fid must not be blank")
        return fid


class AttachmentBody(BaseModel):
    """the JSON body as an object; the endpoint also takes the documents as a bare array"""
    documents: Optional[List[AttachedDocument]] = None


def attached_documents(body):
    """
    the documents of the request body, one per fid in the order given: the first url of a repeated fid wins.

    a miniourl is needed on every document or on none, as in the correspondence service: answering from part of what
    was attached is worse than not answering, because the missing part is invisible, so ValueError (a 422 for the
    backend). no url at all is attaching nothing, and the turn searches the drive.
    """
    documents = body.documents if isinstance(body, AttachmentBody) else body
    unique = {}
    for document in documents or []:
        unique.setdefault(document.fid, document)
    documents = list(unique.values())

    missing = [document.fid for document in documents if not document.miniourl]
    if missing and len(missing) < len(documents):
        raise ValueError(f"a miniourl is needed on every document, or on none: {missing[0]!r} has none, so that file could not be read while the others were")
    if missing:
        logger.warning(f"No document has a miniourl, so nothing is attached --> {missing}")
        return []
    if len(documents) > MAX_ATTACHMENTS:
        raise ValueError(f"at most {MAX_ATTACHMENTS} documents can be attached, got {len(documents)}")
    return documents


def ingest_attachments(attachments, username):
    """
    reads the attached files into ATTACHMENT_INDEX. returns (fIds that can be searched, fIds that could not be read).

    doc_ingest.ingest() is not used because it refuses a document with no tenant_id and this endpoint gets none.
    tenantId stays blank on these chunks: nothing filters on it.
    runs in a worker thread (see service.py): minio, OCR and elasticsearch are all blocking.
    """
    try:
        doc_ingest.ensure_index(ATTACHMENT_INDEX)
    except Exception as e:
        logger.exception(f"Index '{ATTACHMENT_INDEX}' is not usable, attachments not read: {e}")
        return [], [document.fid for document in attachments]

    readable, unread = [], []
    for document in attachments:
        report = doc_ingest.ingest_one(document.fid, document.miniourl, index=ATTACHMENT_INDEX, username=username, file_name=document.fileName)
        logger.info(f"Attachment {document.fid} ({report['fileName']!r}) --> {report['status']}, {report['chunks']} chunk(s) written to '{ATTACHMENT_INDEX}' {report['reason'] or ''}")
        (readable if report["status"] in READABLE_STATUSES else unread).append(document.fid)
    return readable, unread
