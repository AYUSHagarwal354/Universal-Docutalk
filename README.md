# Universal-Docutalk

Universal-Docutalk is a RAG (Retrieval-Augmented Generation) based NLP service designed to provide intelligent document search and conversational capabilities. It leverages **FastAPI** for the backend, **Elasticsearch** for document retrieval (supporting both vector and hybrid search), **Redis** for session management and chat history, and **IBM Cloud/Watson** (LLaMA/Mixtral models) for generating responses.

## Features

- **Hybrid Search**: Supports both vector-based (KNN) and hybrid (RRF - Reciprocal Rank Fusion) retrieval methods.
- **Session Management**: Maintains conversational context using Redis.
- **Multi-Model Support**: Configurable to use different LLMs (LLaMA 3, Mixtral, etc.) via IBM Cloud.
- **Conversation History**:  Stores and retrieves chat history to support context-aware follow-up questions.
- **Streaming Responses**:  Returns generated text in real-time using Server-Sent Events (SSE).
- **Document Attachments**: Files attached to a chat turn are read from MinIO (OCR for scans), indexed apart from the drive corpus, and searched on their own.

## Prerequisites

Ensure you have the following services running and accessible:

- **Python 3.10+**
- **Elasticsearch 8.x** (with vector search capabilities)
- **Redis**
- **IBM Cloud Account** (for LLM access)
- **MinIO** (for attached documents)

## API Documentation

### Search Documents (Chat)
**POST** `/teamsync/nlp/docutalk/search_documents`

Main endpoint for the conversational interface.

**Parameters:**
- `query`: The user's question.
- `username`: User identifier (used for session key).
- `conversationId`: Unique ID for the conversation thread.
- `session`: Boolean. If `true`, maintains history; if `false`, clears it.
- `thinkingRequired`: Boolean.
- `modeltype`: Type of model prompting to use.
- `path`: File path filter for document search.

The parameters above are query parameters. Attached documents go in the optional JSON body, in the shape the correspondence service takes them: a bare array, or the same array under `"documents"`.

```json
[
  {"fid": "abc123", "miniourl": "testamitappolo.comdms//abc123", "fileName": "note.pdf"}
]
```

- `fid`: The document's `fId`.
- `miniourl`: Its MinIO key, shaped `bucket//object`.
- `fileName` (optional): Used when MinIO has no name recorded for the upload.

**Response:**
Returns a `text/event-stream` (SSE) with the generated answer chunks.

**Attachments:**
A turn that sends documents searches only those documents, whatever `scope` and `path` say, and its answer carries no sources and no `meta_data`. A turn that sends none searches the drive as usual, so send them again on every turn that should stay on the attached documents. Each file is read once; the first question on a file waits for the download and OCR before the answer starts.

A `miniourl` is needed on every document or on none: some with and some without, or more than `MAX_ATTACHMENTS` documents, is an HTTP 422, and none at all attaches nothing. A file that cannot be read is reported in the answer; the others are still used.

**Attachment configuration (environment variables):**
For local runs these can sit in a `.env` file anywhere above `app/`; `main.py` loads it at startup and real environment variables win over it.

- `ATTACHMENT_INDEX`: Index attached documents are written to (default `docutalk`). It is created on first use, so the ingest pipeline and synonyms set below must exist on the cluster.
- `MAX_ATTACHMENTS`: Most documents one turn can attach (default `20`).
- `MINIO_ENDPOINT`: Where the attached files are downloaded from (default `http://11.0.0.109:9000`). This must be the S3 API port: 9001 on that host is the browser console and downloads from it fail.
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`: Required. There is no default, so without them every attachment is reported as unreadable.
- `MINIO_REGION`, `MINIO_SOURCE_BUCKET`, `MINIO_DOWNLOAD_RETRIES`, `MINIO_MAX_DOWNLOAD_BYTES`: Optional.
- `EXAM_INGEST_PIPELINE` (default `e5-teamsync-faq-pipeline`), `EXAM_SYNONYMS_SET` (default `syn_set`): As used for `teamsync_v1`.
- `EXAM_OCR_LANGUAGES`: Languages for scans (default `eng`; `eng,hin` to match the correspondence service).

These `EXAM_*` names and the rest of the chunking and OCR settings belong to [app/doc_ingest.py](app/doc_ingest.py), a copy of the correspondence service's module; its `SETTINGS` table lists them all.

### Close Session
**POST** `/teamsync/nlp/docutalk/close_unv_session`

Closes the current session and resets the Redis memory for the user.

**Parameters:**
- `username`: User identifier.
- `conversationId`: Unique ID.
- `session`: Boolean (pass `false` to close).

### Delete Conversation
**POST** `/teamsync/nlp/docutalk/delete_conversation`

Permanently deletes the conversation history from Elasticsearch.

**Parameters:**
- `conversationId`: Unique ID of the conversation to delete.
