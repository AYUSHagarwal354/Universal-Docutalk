"""
    minio -> text -> overlapping chunks -> elasticsearch, in one file

    ONE FUNCTION TO CALL : ingest(). Hand it a document -- one, or a list of
    them -- and it does the whole job.

        from doc_ingest import ingest

        report = ingest({"fid": "abc", "miniourl": "bucket//abc", "tenant_id": "t1"})
        report = ingest([...])       # or a batch

    A document is a plain dict. fid, miniourl and tenant_id are required;
    fileName, username and index are used when present. That is deliberately
    the same shape the team sync new-document message already has, so a
    service holding one can forward it as it stands:

        # a consumer in another service
        import doc_ingest

        for message in source:
            doc = json.loads(message.value)
            if doc.get("index") == "examination":
                report = doc_ingest.ingest(doc)

    A document that names its own "index" routes itself there, so a caller that
    only forwards what it received needs no second argument.

    Self contained on purpose. This module imports nothing from the rest of the
    project, so it can be dropped into another service as it is : it builds its
    own minio and elasticsearch clients and reads its own configuration from the
    environment.

    It is also a script, which is the quickest way to backfill or to see what a
    file that produced a bad paper actually extracted to:

        python doc_ingest.py --fid abc --url bucket//abc --tenant-id t1 --index examination
        python doc_ingest.py --fid abc --url bucket//abc --dry-run

    The steps underneath are importable too, for a service that already has the
    bytes and should not have to go through minio to use the rest:

        extract_chunks(data, file_name)   bytes -> [{pageNo, para, text}]
        ensure_index(index)               create it, mapped like teamsync_v1
        bulk_index(documents, index)      write and wait for searchability

    WHAT IT DOES NOT DO : compute embeddings. The index is created with
    default_pipeline set to the same e5 pipeline teamsync_v1 uses, so
    elasticsearch fills text_embedding in on the way in. That pipeline has to
    exist on the cluster; ensure_index says so plainly if it does not.

    Heavy dependencies -- PyMuPDF, pytesseract, Pillow -- are imported inside
    the functions that need them. Importing this module is cheap, and a
    deployment that only reads docx files never pays for an OCR stack it will
    not use. OCR itself is the tesseract program, run once per scanned page.
"""

import io
import os
import re
import sys
import json
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# ONE OPENMP THREAD PER TESSERACT. Tesseract is built with OpenMP and
# parallelises inside a single page, but it scales badly there -- and this
# module now runs several pages at once, so without this each of those would
# start its own pool and they would fight for the same cores. One thread each,
# several pages at a time, is markedly faster than the other way round.
#
# setdefault, so a deployment that has measured its own hardware can override
# it. Set before any tesseract exists : the pytesseract engine reads it in the
# child process it spawns, and the tesserocr engine reads it when the library
# loads, which is the first time a page is recognised.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

# One table, read at import and again by reload_config() when this file is run
# as a script. Defined once rather than as a run of os.getenv calls plus a
# second copy inside reload_config : two copies is how a default silently drifts
# from the one actually in use, which is exactly what happened when CHUNK_WORDS
# was lowered in one place and not the other.
#
#   env var -> (module attribute, how to read it, default)
#
# A default of None means "unset is a real value here", so the empty string a
# container passes for an unused setting does not become a host name.
SETTINGS = [
    # --- minio ---
    # the S3 API port. 9001 on this host is the browser console, which the client cannot download from
    ("MINIO_ENDPOINT", "MINIO_ENDPOINT", str, "http://11.0.0.109:9000"),
    ("MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY", str, None),
    ("MINIO_SECRET_KEY", "MINIO_SECRET_KEY", str, None),
    ("MINIO_REGION", "MINIO_REGION", str, None),
    # only used when a miniourl names an object but no bucket
    ("MINIO_SOURCE_BUCKET", "MINIO_SOURCE_BUCKET", str, None),
    ("MINIO_DOWNLOAD_RETRIES", "MINIO_DOWNLOAD_RETRIES", int, 2),
    # A source document is read into memory to extract from it. This caps what
    # one oversized file can do to a pod: a 2 GB pdf would otherwise be a 2 GB
    # allocation per concurrent request, and the OOM kill takes every other
    # request with it.
    ("MINIO_MAX_DOWNLOAD_BYTES", "MAX_DOWNLOAD_BYTES", int, 256 * 1024 * 1024),

    # --- elasticsearch ---
    ("ES_HOST", "ES_HOST", str, "elasticsearch-teamsync.apps.lab.ocp.lan"),
    ("ES_PORT", "ES_PORT", int, 80),
    ("SCHEME", "ES_SCHEME", str, "http"),
    ("ELASTIC_USER", "ES_USER", str, "elastic"),
    ("ELASTIC_PASSWORD", "ES_PASSWORD", str, "admin123"),
    ("ES_REQUEST_TIMEOUT", "ES_REQUEST_TIMEOUT", int, 30),

    # --- the index, and how it is built ---
    ("EXAM_TARGET_INDEX", "EXAM_INDEX", str, "examination"),
    # the pipeline that turns `text` into `text_embedding` on the way in : the
    # same one teamsync_v1 uses, so both indices carry vectors from one model
    ("EXAM_INGEST_PIPELINE", "INGEST_PIPELINE", str, "e5-teamsync-faq-pipeline"),
    ("EXAM_INDEX_SHARDS", "INDEX_SHARDS", str, "1"),
    ("EXAM_INDEX_REPLICAS", "INDEX_REPLICAS", str, "1"),
    ("EXAM_SYNONYMS_SET", "SYNONYMS_SET", str, "syn_set"),
    ("EXAM_BULK_CHUNK", "BULK_CHUNK", int, 200),

    # --- chunking ---
    # e5-small truncates at 512 tokens, and a truncated chunk is the quiet
    # failure worth avoiding : its tail is embedded as nothing, so the vector
    # stops representing text the BM25 half still matches on, and hybrid
    # retrieval gets worse with nothing reporting it. Elasticsearch does record
    # it on the document, which is how to check after a backfill:
    #
    #   GET <index>/_count {"query": {"term": {"is_truncated": true}}}
    #
    # 120 measured clean on real documents. 180 did not -- 3 of 24 chunks came
    # back truncated -- because dense prose runs about 10 characters per word
    # and tokenises well above the 1.3 tokens/word a round English average
    # suggests.
    ("EXAM_CHUNK_WORDS", "CHUNK_WORDS", int, 120),
    # The word count alone is not enough. A table row or a run of long
    # identifiers has few words and many tokens, so a window that is short by
    # words can still blow the limit; this is the backstop that catches it.
    # ~1200 characters is about 400 tokens at the ~3 chars/token these
    # documents average.
    ("EXAM_MAX_CHUNK_CHARS", "MAX_CHUNK_CHARS", int, 1200),
    # carried from the end of one chunk into the start of the next, so a
    # sentence split across the boundary is still whole in one of them
    ("EXAM_CHUNK_OVERLAP_WORDS", "CHUNK_OVERLAP_WORDS", int, 30),
    # below this a chunk is a page number or a running header, not material a
    # question can be written from
    ("EXAM_MIN_CHUNK_WORDS", "MIN_CHUNK_WORDS", int, 15),

    # --- extraction ---
    # Below this many characters a pdf page's text layer is treated as absent
    # and the page is OCR'd. A scanned page is not empty -- it usually carries a
    # few characters of junk from the scanner -- so "any text at all" is not the
    # test.
    ("EXAM_MIN_PAGE_TEXT_CHARS", "MIN_PAGE_TEXT_CHARS", int, 60),
    # a pdf page is 72dpi at scale 1, so 2.5 renders at 180dpi. Measured on a
    # real scanned letter, rendered by PyMuPDF : at 180dpi tesseract read the
    # date, the address, the signatory and the e-mail address all correctly,
    # at 216 and 252dpi it garbled the e-mail -- that scan is ~94dpi, and
    # enlarging a scan well past its own resolution adds blur, not detail --
    # and 180dpi was the fastest of the three. An A4 page is then about
    # 1500 x 2100 pixels
    ("EXAM_OCR_RENDER_SCALE", "OCR_RENDER_SCALE", float, 2.5),
    # the tesseract program, when it is not on PATH. Empty looks on PATH, then
    # where the Windows installer puts it
    ("TESSERACT_CMD", "TESSERACT_CMD", str, ""),
    # the share of a pdf page one image has to cover before the page counts as
    # having a picture that could hold text. 0.01 is about a 2.5 cm square on A4 :
    # a scanned page covers all of it, a bullet or a rule none of it
    ("EXAM_OCR_MIN_IMAGE_AREA", "OCR_MIN_IMAGE_AREA", float, 0.01),
    # how many pages of one document are recognised at once. 0 picks from the
    # cpu count, capped at 4 -- each worker is a whole tesseract, and on an A4
    # page at 180dpi that is hundreds of megabytes, so a pod with a memory
    # limit wants this pinned rather than guessed. 1 is the old behaviour,
    # one page after another.
    ("EXAM_OCR_WORKERS", "OCR_WORKERS", int, 0),
    # which engine recognises a page : "auto" prefers tesserocr and falls back
    # to pytesseract, "pytesseract" forces the subprocess engine -- the way
    # back to one tesseract per page, for a deployment that wants a page read
    # in a process of its own rather than in this one
    ("EXAM_OCR_ENGINE", "OCR_ENGINE", str, "auto"),
]

# Every attribute _apply_settings() binds, declared once here as well.
#
# The value below is never the one in use -- _apply_settings() overwrites all of
# them from SETTINGS a few lines down, before anything reads them -- so this is
# a declaration, not a second table of defaults to drift out of step with the
# first. It exists because Cython binds module level names at compile time and
# cannot see a name that only ever appears as a setattr target: without it the
# protected build fails with "undeclared name not builtin: MINIO_ENDPOINT" and
# one line per setting after it. Interpreted, it costs nothing.
#
# A setting added to SETTINGS needs its attribute added here too.
MINIO_ENDPOINT = None
MINIO_ACCESS_KEY = None
MINIO_SECRET_KEY = None
MINIO_REGION = None
MINIO_SOURCE_BUCKET = None
MINIO_DOWNLOAD_RETRIES = None
MAX_DOWNLOAD_BYTES = None

ES_HOST = None
ES_PORT = None
ES_SCHEME = None
ES_USER = None
ES_PASSWORD = None
ES_REQUEST_TIMEOUT = None

EXAM_INDEX = None
INGEST_PIPELINE = None
INDEX_SHARDS = None
INDEX_REPLICAS = None
SYNONYMS_SET = None
BULK_CHUNK = None

CHUNK_WORDS = None
MAX_CHUNK_CHARS = None
CHUNK_OVERLAP_WORDS = None
MIN_CHUNK_WORDS = None

MIN_PAGE_TEXT_CHARS = None
OCR_RENDER_SCALE = None
TESSERACT_CMD = None
OCR_MIN_IMAGE_AREA = None
OCR_WORKERS = None
OCR_ENGINE = None
OCR_LANGUAGES = None
OCR_ENABLED = None


def _apply_settings():
    """Binds every entry in SETTINGS to this module from the environment."""
    module = sys.modules[__name__]
    for variable, attribute, caster, default in SETTINGS:
        raw = os.getenv(variable)
        if raw is None or (raw == "" and default is None):
            value = None if default is None else default
        else:
            try:
                value = caster(raw)
            except (TypeError, ValueError):
                logger.warning(
                    f"{variable}={raw!r} is not a valid {caster.__name__}, "
                    f"using {default!r}"
                )
                value = default
        setattr(module, attribute, value)

    # the two that are not a plain cast. Languages as tesseract names them
    # ("eng", "hin") or as ISO 639-1 codes ("en", "hi") -- see
    # tesseract_languages
    module.OCR_LANGUAGES = [lang.strip()
                            for lang in os.getenv("EXAM_OCR_LANGUAGES", "eng").split(",")
                            if lang.strip()]
    module.OCR_ENABLED = str(os.getenv("EXAM_OCR_ENABLED", "true")).strip().lower() == "true"


_apply_settings()


class ExtractionError(Exception):
    """the file could not be read at all, as opposed to having no text in it"""


# ----------------------------------------------------------------------------
# clients
# ----------------------------------------------------------------------------

# Built on first use, not at import. Importing this module must not open a
# socket or fail because a dependency is down, since other services import it
# for extract_chunks alone and never touch minio or elasticsearch.
_minio_client = None
_es_client = None


def parse_endpoint(endpoint):
    """
    The minio client wants a bare host[:port] plus a secure flag, but a full URL
    is what people actually have to hand. Accept either.
    """
    from urllib.parse import urlparse

    endpoint = (endpoint or "").strip()
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        return parsed.netloc.strip("/"), parsed.scheme == "https"
    return endpoint.strip("/"), False


def get_minio_client():
    """the shared minio client, built once"""
    global _minio_client
    if _minio_client is not None:
        return _minio_client

    from minio import Minio

    host, secure = parse_endpoint(MINIO_ENDPOINT)
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        logger.warning(
            "MINIO_ACCESS_KEY / MINIO_SECRET_KEY are not set, downloads will fail"
        )
    _minio_client = Minio(
        host,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=secure,
        region=MINIO_REGION,
    )
    logger.info(f"doc_ingest minio client : host={host} secure={secure}")
    return _minio_client


def get_es_client():
    """the shared elasticsearch client, built once"""
    global _es_client
    if _es_client is not None:
        return _es_client

    from elasticsearch import Elasticsearch

    _es_client = Elasticsearch(
        [{"host": ES_HOST, "port": ES_PORT, "scheme": ES_SCHEME}],
        basic_auth=(ES_USER, ES_PASSWORD),
        request_timeout=ES_REQUEST_TIMEOUT,
    )
    logger.info(f"doc_ingest elasticsearch client : {ES_SCHEME}://{ES_HOST}:{ES_PORT}")
    return _es_client


# ----------------------------------------------------------------------------
# minio
# ----------------------------------------------------------------------------

def split_object_key(key, default_bucket=None):
    """
    Splits a "bucket/path/to/object" key into (bucket, object_name).

    This is the shape miniourl arrives in, and it is not tidy:

        testamitappolo.comdms//6a96aa42882b2823e6061b97
        testamitappolo.comdms//67ea7b3f.../67ea7c76.../67ea7ca8...

    The doubled slash is the bucket followed by a path that already begins with
    one. Leading slashes are stripped from the object name because s3 object
    names do not carry them -- "/a/b" and "a/b" are two different keys and only
    the second is the one that was written.
    """
    key = (key or "").strip()
    if not key:
        raise ValueError("the minio url is empty")

    # a full URL carries a scheme and host that say nothing about which bucket
    # the object is in, so it is refused rather than guessed at
    if "://" in key:
        raise ValueError(
            f"expected a bucket qualified key like 'bucket/path/object', got a URL: {key!r}"
        )

    bucket, separator, object_name = key.lstrip("/").partition("/")
    if not separator:
        # no slash at all: the whole thing is an object name, which is how a
        # caller that only knows the file id sends it
        if not default_bucket:
            raise ValueError(f"{key!r} names no bucket and MINIO_SOURCE_BUCKET is not set")
        return default_bucket, bucket

    object_name = object_name.lstrip("/")
    if not object_name:
        raise ValueError(f"{key!r} names a bucket but no object")
    return bucket, object_name


def original_name(stat):
    """
    The upload's original file name, if whoever wrote the object recorded one.

    Worth asking for: these objects are named by document id, and a question
    attributed to "6a96aa42882b2823e6061b97" is not a citation anyone can act
    on. s3 user metadata reaches the client with an "x-amz-meta-" prefix and
    minio normalises the case inconsistently between versions, so the keys are
    matched case insensitively rather than looked up.
    """
    from urllib.parse import unquote

    metadata = getattr(stat, "metadata", None) or {}
    wanted = ("x-amz-meta-filename", "x-amz-meta-file-name",
              "x-amz-meta-original-filename", "filename", "file-name")
    for key, value in metadata.items():
        if str(key).strip().lower() in wanted and str(value).strip():
            return str(value).strip()

    # content-disposition is the other place a name is conventionally kept
    disposition = str(getattr(stat, "content_disposition", "") or "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip())
    return ""


def download(key, client=None):
    """
    Reads one object named by a bucket qualified key, returning (bytes, file_name).

    Into memory rather than to a temp file: the extractor works on bytes, and a
    pod that writes every source document to disk needs a writable volume sized
    for the largest burst it will ever see. MAX_DOWNLOAD_BYTES is what keeps
    that decision safe.
    """
    client = client or get_minio_client()
    bucket, object_name = split_object_key(key, default_bucket=MINIO_SOURCE_BUCKET)
    last_error = None

    for attempt in range(1, MINIO_DOWNLOAD_RETRIES + 1):
        start = time.time()
        response = None
        try:
            stat = client.stat_object(bucket, object_name)
            size = getattr(stat, "size", 0) or 0
            if size > MAX_DOWNLOAD_BYTES:
                # not retried: it will be too large next time too
                raise ValueError(
                    f"'{key}' is {size} bytes, over the {MAX_DOWNLOAD_BYTES} byte limit"
                )

            response = client.get_object(bucket, object_name)
            data = response.read()
            name = original_name(stat) or object_name.rstrip("/").split("/")[-1]
            logger.info(
                f"Downloaded {bucket}/{object_name} ({len(data)} bytes, "
                f"name={name!r}) in {time.time() - start:.2f}s"
            )
            return data, name
        except ValueError:
            raise
        except Exception as e:
            last_error = e
            logger.warning(
                f"MinIO download attempt {attempt}/{MINIO_DOWNLOAD_RETRIES} for "
                f"'{bucket}/{object_name}' failed after {time.time() - start:.2f}s : "
                f"{type(e).__name__}: {e}"
            )
        finally:
            # get_object hands back a live urllib3 response; not closing it leaks
            # the connection out of the pool, and the pool is small
            if response is not None:
                try:
                    response.close()
                    response.release_conn()
                except Exception:
                    pass

    raise RuntimeError(
        f"MinIO download of '{key}' failed after {MINIO_DOWNLOAD_RETRIES} attempts: {last_error}"
    )


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
#
# Tesseract, run as its own program through pytesseract. Nothing is loaded into
# this process : each page is one call to the tesseract binary, which reads the
# language data from its own tessdata folder. So there is no model to warm up,
# no gpu to find, and no weights to download -- the languages are whatever
# tessdata holds, installed with tesseract itself.

# The ISO 639-1 codes EXAM_OCR_LANGUAGES used to take, mapped to tesseract's own
# (ISO 639-2) names, so a config written as "en,hi" keeps working. A code
# already in tesseract's form -- "eng", "hin", "script/Devanagari" -- passes
# through untouched.
TESSERACT_LANGUAGE_CODES = {
    "en": "eng", "hi": "hin", "mr": "mar", "bn": "ben", "gu": "guj",
    "pa": "pan", "ta": "tam", "te": "tel", "kn": "kan", "ml": "mal",
    "or": "ori", "ur": "urd", "ne": "nep", "sa": "san",
}

# where the Windows installer puts tesseract, when it is not on PATH -- the
# installer does not add it by default
_WINDOWS_TESSERACT = os.path.join(
    os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe")

# Resolved once and reused : which binary, and the "eng+hin" string passed as
# lang. None when OCR is unavailable; the failure is remembered, so a missing
# binary is not looked for and re-logged once per page.
_tesseract_lang = None
_ocr_failed = False


def tesseract_languages(configured):
    """
    EXAM_OCR_LANGUAGES as tesseract names them : ["en", "hi"] -> ["eng", "hin"].
    Order is kept -- tesseract tries the first language first -- and a repeat
    is dropped.
    """
    languages = []
    for code in configured:
        name = TESSERACT_LANGUAGE_CODES.get(code.lower(), code)
        if name not in languages:
            languages.append(name)
    return languages


def ocr_available():
    """
    Whether scanned pages can be read, set up on first use : the tesseract
    binary found, and the configured languages checked against what its
    tessdata holds.

    False rather than an exception : a deployment without tesseract should
    still ingest born-digital pdfs and docx files perfectly well, just not
    scans. Losing scanned pages is worth a loud warning; it is not worth
    failing every ingestion.

    A configured language tesseract does not have is dropped with a warning
    rather than passed on. Tesseract refuses to run at all when one language
    in "eng+hin" is missing, so passing it would turn a Hindi pack that was
    never installed into English scans that cannot be read either.
    """
    global _tesseract_lang, _ocr_failed
    if _tesseract_lang is not None or _ocr_failed:
        return _tesseract_lang is not None

    if not OCR_ENABLED:
        logger.info("OCR is disabled by EXAM_OCR_ENABLED, scanned pages will be skipped")
        _ocr_failed = True
        return False

    try:
        import pytesseract
    except Exception as e:
        logger.warning(
            f"pytesseract is not installed ({e}) : pages with no text layer will be "
            f"skipped. pip install pytesseract to read scanned documents."
        )
        _ocr_failed = True
        return False

    import shutil
    command = TESSERACT_CMD or shutil.which("tesseract")
    if not command and os.path.isfile(_WINDOWS_TESSERACT):
        command = _WINDOWS_TESSERACT
    if not command:
        logger.warning(
            "The tesseract program was not found (not on PATH, and TESSERACT_CMD is "
            "not set) : pages with no text layer will be skipped. Install tesseract, "
            "or set TESSERACT_CMD to its full path, to read scanned documents."
        )
        _ocr_failed = True
        return False
    pytesseract.pytesseract.tesseract_cmd = command

    try:
        version = pytesseract.get_tesseract_version()
        installed = set(pytesseract.get_languages(config=""))
    except Exception as e:
        logger.warning(
            f"The tesseract program at {command} could not be run "
            f"({type(e).__name__}: {e}) : pages with no text layer will be skipped"
        )
        _ocr_failed = True
        return False

    wanted = tesseract_languages(OCR_LANGUAGES)
    usable = [name for name in wanted if name in installed]
    missing = [name for name in wanted if name not in installed]
    if missing:
        logger.warning(
            f"OCR : tesseract has no language data for {missing} (EXAM_OCR_LANGUAGES="
            f"{','.join(OCR_LANGUAGES)}), so scans are read without "
            f"{'it' if len(missing) == 1 else 'them'}. Installed : {sorted(installed)}. "
            f"Add the .traineddata file(s) to tesseract's tessdata folder to read them."
        )
    if not usable:
        logger.warning(
            f"OCR : none of the configured languages {wanted} is installed in "
            f"tesseract, so pages with no text layer will be skipped"
        )
        _ocr_failed = True
        return False

    _tesseract_lang = "+".join(usable)
    logger.info(f"OCR ready : tesseract {version} at {command}, languages={_tesseract_lang}")
    return True


# TWO ENGINES, ONE TESSERACT. Both run the same tesseract and the same
# language data; they differ only in how a page reaches it.
#
#   tesserocr     the C API, in this process. The language data is loaded once
#                 per worker thread and reused for every page that thread
#                 handles, and the image is passed in memory. Preferred,
#                 because a letter is mostly fixed cost : spawning a process,
#                 loading tessdata and writing a temp image for every page adds
#                 up to a large share of a short page's time.
#   pytesseract   the tesseract program, one child process per page. Always
#                 available, being a pure python wrapper, and the page is read
#                 in a process of its own : one that dies takes nothing of this
#                 service with it.
#
# The fallback is not a formality : tesserocr is a compiled extension, and an
# image that could not build it must still read scans.
#
# NEITHER GIVES UP ON A PAGE. There is no per-page time limit: tesseract is
# left to finish whatever it was given, and a page that hangs it holds its
# worker until it does. The limit that used to be here cut off slow pages as
# well as stuck ones -- a dense or badly scanned page is slow and still
# readable, and losing it silently put XXX in a letter where the document had
# an answer all along.
_local = threading.local()
_tesserocr_unusable = False


def _tesserocr_api():
    """
    This thread's tesseract, or None when tesserocr cannot be used.

    One PyTessBaseAPI per thread, never shared : the C API holds the image and
    the result of the page it is working on, so two threads in one instance
    read each other's page. Held on a thread local so every page that thread
    goes on to recognise reuses the language data already loaded into it.
    """
    global _tesserocr_unusable
    if _tesserocr_unusable or (OCR_ENGINE or "auto").strip().lower() == "pytesseract":
        return None

    api = getattr(_local, "tesserocr_api", None)
    if api is not None:
        return api

    try:
        import tesserocr
        api = tesserocr.PyTessBaseAPI(lang=_tesseract_lang, psm=tesserocr.PSM.AUTO)
    except Exception as e:
        # once, for the process : a missing extension is not going to appear
        # between one page and the next, and saying so per page would bury the
        # log of a long scan
        _tesserocr_unusable = True
        logger.info(
            f"tesserocr is not usable ({type(e).__name__}: {e}), recognising pages "
            f"with pytesseract instead -- one tesseract process per page"
        )
        return None

    _local.tesserocr_api = api
    return api


def _end_tesserocr_api():
    """Releases this thread's tesseract, if it has one."""
    api = getattr(_local, "tesserocr_api", None)
    if api is None:
        return
    _local.tesserocr_api = None
    try:
        api.End()
    except Exception as e:
        logger.debug(f"A tesserocr instance did not close cleanly: {e}")


def _recognise(image):
    """
    One PIL image -> the text on it, through whichever engine is available.

    psm 3 in both : find the blocks on the page itself, which is what a letter
    with a letterhead, an address block and a signature needs.

    Neither engine is given a deadline -- see the note above on the two of
    them. A page takes as long as it takes.
    """
    api = _tesserocr_api()
    if api is not None:
        api.SetImage(image)
        return api.GetUTF8Text() or ""

    import pytesseract
    return pytesseract.image_to_string(image, lang=_tesseract_lang, config="--psm 3")


def ocr_image(image_bytes):
    """
    Recognises one page image -- a rendered pdf page, or an uploaded photo or
    scan.

    Returns "" when OCR is unavailable or finds nothing, and never raises: one
    unreadable page should cost that page, not the document. That holds inside
    a worker thread too, where a raise would otherwise surface as one page's
    exception standing in for the whole document's text.
    """
    if not ocr_available():
        return ""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as image:
            text = _recognise(image)
        lines = (line.rstrip() for line in text.splitlines())
        return "\n".join(line for line in lines if line.strip())
    except Exception as e:
        logger.warning(f"OCR failed on a page: {type(e).__name__}: {e}")
        return ""


def ocr_workers():
    """
    How many pages of one document to recognise at once.

    Capped at 4 by default rather than the whole cpu count : each worker holds
    a tesseract and a rendered A4 page, and a pod that fans out to sixteen of
    them is killed for memory long before it runs out of cores.
    """
    if OCR_WORKERS and OCR_WORKERS > 0:
        return OCR_WORKERS
    return max(1, min(4, os.cpu_count() or 1))


def recognise_pages(images):
    """
    [image bytes] -> [text], in the same order.

    One page is recognised on this thread : a pool for a single page is pure
    overhead, and it keeps the common case -- an uploaded photo, a one page
    scan -- on exactly the path it was always on.

    Several pages are shared across a pool, which is where the time goes on a
    real scan. Each worker ends its own tesseract when the pool closes, so a
    document does not leave one behind; the language data is therefore loaded
    once per worker per document rather than once per page.
    """
    if not images:
        return []

    workers = min(ocr_workers(), len(images))
    if workers <= 1:
        return [ocr_image(image) for image in images]

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="ocr") as pool:
        try:
            texts = list(pool.map(ocr_image, images))
        finally:
            # inside the pool, so each worker releases the tesseract it built
            for _ in range(workers):
                pool.submit(_end_tesserocr_api)
    logger.info(
        f"OCR : {len(images)} page(s) on {workers} worker(s) in {time.time() - start:.2f}s"
    )
    return texts


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------

def extract_pdf(data):
    """
    Returns [(page_number, text)], page numbers starting at 1.

    PyMuPDF reads every page, and each page is decided on its own, cheapest
    first:

        no image on it           its text layer, as PyMuPDF reads it. Never
                                 OCR'd : with no picture on the page there is
                                 nothing for OCR to find that the text layer
                                 does not already hold. A blank or near blank
                                 page used to cost a render and a tesseract run
                                 for nothing.
        an image, and text       its text layer. The image is a letterhead
                                 emblem, a signature or a stamp -- or the page
                                 is a searchable scan, whose scanner already put
                                 the text on it.
        an image, and no text    OCR'd : a scanned page.

    "An image" is one covering at least OCR_MIN_IMAGE_AREA of the page, so a
    bullet, a rule or a spacer pixel does not count. "No text" is fewer than
    MIN_PAGE_TEXT_CHARS characters : a scanned page usually carries a few
    characters of junk, so "none at all" is not the test.

    The text layer wins wherever there is one because it is what the document
    actually says, and it costs milliseconds. OCR is a guess at a picture of
    it, costs seconds a page, and on a page with real text is strictly worse --
    it mangles tables and invents characters that then end up in a letter.

    A page that is empty either way is still returned, as an empty string, so
    page numbers stay the document's own.
    """
    try:
        import pymupdf
    except Exception as e:
        raise ExtractionError(f"PyMuPDF is required to read pdfs: {e}")

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ExtractionError(f"the pdf could not be opened: {type(e).__name__}: {e}")

    pages = []
    # how each page was read, for the one line that says where the time went
    read = {"text": 0, "ocr": 0, "no_image": 0, "unread": 0}
    # the scans, rendered and waiting to be recognised : [(slot in pages, png)].
    # Rendering stays on this thread -- it is milliseconds, and a pymupdf page
    # belongs to the document it was opened from -- and only recognition, which
    # is the part measured in seconds, is shared out.
    scans = []
    start = time.time()
    try:
        for page in document:
            try:
                text = page.get_text("text") or ""
            except Exception as e:
                logger.debug(f"Page {page.number + 1} has no readable text layer: {e}")
                text = ""

            if len(text.strip()) >= MIN_PAGE_TEXT_CHARS:
                read["text"] += 1
            elif not has_image(page):
                # little or no text, and no picture that could hold more
                read["no_image"] += 1
            else:
                # a scan. Rendered only when it can be read : a page image
                # nobody can OCR costs seconds and memory for nothing
                rendered = render_page(page) if ocr_available() else None
                if rendered:
                    scans.append((len(pages), rendered))
                else:
                    read["unread"] += 1

            pages.append([page.number + 1, text])
    finally:
        # MuPDF holds the document's buffers natively until it is closed
        try:
            document.close()
        except Exception:
            pass

    # every scan in this document at once, the document's own pages closed
    # behind us : the images are bytes now and hold nothing of mupdf's
    for (slot, _), recognised in zip(scans, recognise_pages([png for _, png in scans])):
        if recognised.strip():
            pages[slot][1] = recognised
            read["ocr"] += 1
        else:
            read["unread"] += 1

    pages = [(number, clean_text(text)) for number, text in pages]

    logger.info(
        f"PDF read in {time.time() - start:.2f}s : {len(pages)} page(s) -- "
        f"{read['text']} from the text layer, {read['ocr']} OCR'd, "
        f"{read['no_image']} with little or no text and no image (not OCR'd)"
        + (f", {read['unread']} scanned page(s) OCR could not read" if read["unread"] else "")
    )
    return pages


def has_image(page):
    """
    Whether a picture big enough to hold text is drawn on the page : an image
    covering at least OCR_MIN_IMAGE_AREA of it.

    get_image_info rather than get_images. get_images lists what the page's
    resources make available, and many generators share one resource
    dictionary across every page -- each page would then "have" every image
    in the file. get_image_info reports what is actually drawn on this page,
    inline images included, and where.

    A page whose images cannot be listed is treated as having one : it is only
    asked about a page with no text, and OCR'ing a page that turns out blank
    costs seconds, while skipping a scan loses it.
    """
    try:
        import pymupdf
        area = page.rect.width * page.rect.height
        if area <= 0:
            return False
        for info in page.get_image_info():
            # clipped to the page : an image may run off its edge
            shown = pymupdf.Rect(info["bbox"]) & page.rect
            if shown.width * shown.height >= OCR_MIN_IMAGE_AREA * area:
                return True
        return False
    except Exception as e:
        logger.debug(f"Page {page.number + 1} : images could not be listed ({e}), assuming a scan")
        return True


def render_page(page):
    """
    One pdf page as a greyscale png, or None if it could not be rendered.

    Greyscale because tesseract reads grey anyway, and a third of the pixels
    renders and encodes faster.
    """
    try:
        import pymupdf
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE),
            colorspace=pymupdf.csGRAY, alpha=False,
        )
        return pixmap.tobytes("png")
    except Exception as e:
        logger.warning(f"A pdf page could not be rendered for OCR: {type(e).__name__}: {e}")
        return None


def extract_docx(data):
    """
    Returns [(1, text)] : a docx has no fixed pages, so everything is page 1.

    Reporting a made up page number would be worse than reporting one that is
    honestly meaningless -- pageNo is cited back to the user on every question.

    Table cells are included. In a course handout the tables are often the part
    most worth asking questions about.
    """
    try:
        import docx
    except Exception as e:
        raise ExtractionError(f"python-docx is required to read docx files: {e}")

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ExtractionError(f"the docx could not be opened: {type(e).__name__}: {e}")

    blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                # pipe separated, so a row stays one line and the model can still
                # see where one cell ends and the next begins. Not a tab: clean_text
                # collapses tabs into spaces -- rightly, they are noise in a pdf
                # text layer -- which would erase the cell boundary entirely.
                blocks.append(" | ".join(cells))

    return [(1, clean_text("\n".join(blocks)))]


def extract_text_file(data):
    """Plain text, for the odd .txt or .md that gets uploaded with the rest."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return [(1, clean_text(data.decode(encoding)))]
        except UnicodeDecodeError:
            continue
    return [(1, clean_text(data.decode("utf-8", errors="replace")))]


def extract_image(data):
    """An uploaded photo or scan is OCR only: there is no text layer to try."""
    return [(1, clean_text(ocr_image(data)))]


PDF_MAGIC = b"%PDF"
# a docx is a zip, but so is every other ooxml file, hence the extension check
# first and this only as a fallback
ZIP_MAGIC = b"PK\x03\x04"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".log")


def extract(data, file_name=""):
    """
    Extraction dispatched on what the bytes actually are, with the file name
    only as a tiebreak.

    The name is not trusted first because these files arrive named by their
    document id -- "6a96aa42882b2823e6061b97", no extension at all -- so on the
    path that matters most there is nothing in the name to dispatch on.
    """
    if not data:
        raise ExtractionError("the file is empty")

    name = (file_name or "").lower()

    if data[:4] == PDF_MAGIC or name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    if name.endswith(IMAGE_EXTENSIONS):
        return extract_image(data)
    if name.endswith(TEXT_EXTENSIONS):
        return extract_text_file(data)
    if data[:4] == ZIP_MAGIC:
        # unnamed ooxml: docx is the only kind this module reads, and
        # extract_docx raises a clear ExtractionError if it is an xlsx
        return extract_docx(data)

    # an unnamed file that is neither pdf nor zip. An image is the common case
    # (a phone photo of a page), and OCR on something that is not an image comes
    # back empty rather than raising
    return extract_image(data)


# ----------------------------------------------------------------------------
# chunking
# ----------------------------------------------------------------------------

def clean_text(text):
    """
    Collapses the whitespace pdf and OCR output is full of, without joining
    lines into one another.

    Blank lines are kept as paragraph breaks: by this point they are the only
    structure left, and a chunk that starts mid-paragraph reads worse to the
    model than one that starts at a break.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # a pdf text layer routinely has runs of spaces between glyphs on a line
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=None, overlap=None):
    """
    Cuts one page's text into overlapping word windows.

    Words rather than characters: a character window cuts through the middle of
    a word, and the fragment left on each side is noise in both the embedding
    and the BM25 index.

    The overlap is what stops a definition that straddles a boundary from being
    unanswerable. Without it "the three way handshake consists of" can end one
    chunk and "SYN, SYN-ACK and ACK" start the next, and no single chunk can
    produce that question.
    """
    size = size or CHUNK_WORDS
    overlap = CHUNK_OVERLAP_WORDS if overlap is None else overlap

    # an overlap at or above the size would never advance, so the loop below
    # would not terminate
    if overlap >= size:
        logger.warning(
            f"chunk overlap {overlap} >= size {size}, using {max(size // 4, 1)} instead"
        )
        overlap = max(size // 4, 1)

    words = (text or "").split()
    if not words:
        return []

    chunks = []
    start = 0
    total = len(words)

    while start < total:
        # grow the window a word at a time, stopping at whichever limit comes
        # first : the word count, or the character budget that stands in for the
        # model's token limit
        end = start
        length = 0
        while end < total and (end - start) < size:
            addition = len(words[end]) + (1 if end > start else 0)
            if length + addition > MAX_CHUNK_CHARS and end > start:
                break
            length += addition
            end += 1

        # a single word longer than the whole budget would otherwise never
        # advance the window
        if end == start:
            end = start + 1

        window = words[start:end]
        # the tail of a page is usually a footer, but it can also be the last
        # real paragraph, so it is kept when it is the only chunk there is
        if len(window) < MIN_CHUNK_WORDS and chunks:
            break

        chunks.append(" ".join(window))
        if end >= total:
            break

        # the overlap is measured against the window actually taken, not the
        # requested size : a window cut short by the character budget would
        # otherwise step backwards past where it began
        step = max(len(window) - overlap, 1)
        start += step

    return chunks


def chunk_pages(pages, size=None, overlap=None):
    """
    Every page's chunks, as [{"pageNo": int, "para": int, "text": str}].

    Chunks never span a page, so a question generated from one has a single page
    to cite. para is the chunk's position within its page, which makes a
    document id built from (fId, pageNo, para) stable across re-ingestion.
    """
    chunks = []
    for page_no, text in pages:
        for index, chunk in enumerate(chunk_text(text, size=size, overlap=overlap)):
            chunks.append({"pageNo": page_no, "para": index, "text": chunk})
    return chunks


def extract_chunks(data, file_name=""):
    """extract -> chunk : bytes in, indexable chunks out, no i/o of any kind"""
    pages = extract(data, file_name=file_name)
    chunks = chunk_pages(pages)
    filled = sum(1 for _, text in pages if text.strip())
    logger.info(
        f"Extracted {file_name or 'document'} : {len(pages)} page(s), "
        f"{filled} with text, {len(chunks)} chunk(s)"
    )
    return chunks


# ----------------------------------------------------------------------------
# elasticsearch
# ----------------------------------------------------------------------------

def index_body():
    """
    The target index, created exactly the way the ingestion service creates
    teamsync_v1.

    Same creation body on purpose, not similar: one retrieval query runs against
    whichever index doc_mode picked, so anything that differs here changes the
    results depending on which one that was. The parts that carry that risk:

        syn_search_analyzer  on text and fileName. Without the identical
                             analyzer chain, BM25 scores are not comparable
                             between the two indices, and the RRF half that
                             uses them ranks differently for no visible reason.
        fId / in_trash       the two fields the exam filter is built from
        default_pipeline     what fills text_embedding in, since nothing in this
                             module computes a vector
        _routing required    every write has to name one; this module routes by
                             fId, which keeps a document's chunks on one shard

    NOTE text_embedding is deliberately NOT declared. The ingest pipeline
    produces it and elasticsearch maps it dynamically on the first document --
    a float array of 128..4096 dims becomes a dense_vector with cosine
    similarity and int8_hnsw index options. That is where teamsync_v1's own
    "dims": 384 mapping came from: it is what a GET reports back after the fact,
    not something the create asked for. Declaring it here would be this service
    guessing at a shape the pipeline already decides, and a guess that came out
    differently would change knn scores between the two indices.

    The read-only settings elasticsearch reports back on an existing index --
    uuid, creation_date, provided_name, version -- are absent for the same
    reason: the cluster assigns them and rejects a create that sets them.
    """
    return {
        "settings": {
            # a sibling of "index" rather than inside it. Elasticsearch accepts
            # either -- the "index." prefix is implied -- and this is the shape
            # the ingestion service already sends.
            "analysis": {
                "filter": {
                    "synonyms_filter": {
                        "type": "synonym_graph",
                        "synonyms_set": SYNONYMS_SET,
                        "updateable": True,
                    }
                },
                "analyzer": {
                    "syn_search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "synonyms_filter"],
                    }
                },
            },
            "index": {
                "number_of_shards": INDEX_SHARDS,
                "number_of_replicas": INDEX_REPLICAS,
                "default_pipeline": INGEST_PIPELINE,
            },
        },
        "mappings": {
            "_routing": {"required": True},
            "properties": index_properties(),
        },
    }


def index_properties():
    """
    The mapped fields, and the whole set of them: build_documents writes these
    and nothing else.

    Writing a field that is not here is not an error -- dynamic mapping would
    accept it -- but it would put a field in this index that teamsync_v1 does
    not have, on an index whose entire purpose is to behave like teamsync_v1.
    """
    return {
        "text": {
            "type": "text",
            "search_analyzer": "syn_search_analyzer",
        },
        "fId": {
            "type": "keyword",
            "fields": {"text": {"type": "text"}},
        },
        "fileName": {
            "type": "text",
            "search_analyzer": "syn_search_analyzer",
        },
        "pageNo": {"type": "integer"},
        "tables": {"type": "keyword"},
        "para": {"type": "integer"},
        "username": {
            "type": "keyword",
            "fields": {"text": {"type": "text"}},
        },
        "path": {
            "type": "keyword",
            "fields": {"text": {"type": "text"}},
        },
        "in_trash": {"type": "boolean"},
        "shared_with": {"type": "keyword"},
        "priority": {"type": "keyword"},
        # recorded on every chunk and published with the result; nothing
        # queries it. Mapped exactly as teamsync_v1 maps it -- text with a
        # keyword subfield -- so both indices keep the same shape.
        "tenantId": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
    }


# Fields added to the mapping after the index was first created. create only
# runs on an index that is absent, so an index that predates them gets them
# through put_mapping instead -- adding a field to an existing mapping is
# non-breaking.
ADDED_FIELDS = ("tenantId",)
_fields_ensured = set()


def ensure_added_fields(index, client):
    """
    Adds ADDED_FIELDS to an existing index's mapping, once per index per process.

    Not fatal when it fails. Dynamic mapping would give tenantId exactly the
    text + keyword(256) shape declared here -- that is where teamsync_v1's own
    came from -- and nothing queries the field anyway.
    """
    if index in _fields_ensured:
        return
    properties = index_properties()
    try:
        client.indices.put_mapping(
            index=index,
            body={"properties": {name: properties[name] for name in ADDED_FIELDS}},
        )
        _fields_ensured.add(index)
        logger.info(f"Index '{index}' mapping carries {', '.join(ADDED_FIELDS)}")
    except Exception as e:
        logger.warning(f"Could not add {', '.join(ADDED_FIELDS)} to '{index}': {e}")


def ensure_index(index=None, client=None):
    """
    Creates the index if it is not there, and does nothing if it is.

    On first use rather than at startup: a deployment that never ingests locally
    should not create an index it will not write to, and a cluster that is down
    at boot should not stop a service from coming up.

    A race between two concurrent first requests is expected and handled -- both
    check, both create, one gets resource_already_exists_exception, which is
    success as far as the caller is concerned.
    """
    index = index or EXAM_INDEX
    client = client or get_es_client()

    try:
        if client.indices.exists(index=index):
            ensure_added_fields(index, client)
            return False
    except Exception as e:
        logger.warning(f"Could not check whether '{index}' exists: {e}")

    try:
        client.indices.create(index=index, body=index_body())
        _fields_ensured.add(index)
        logger.info(
            f"Created index '{index}' : shards={INDEX_SHARDS} "
            f"replicas={INDEX_REPLICAS} pipeline={INGEST_PIPELINE}"
        )
        return True
    except Exception as e:
        detail = str(e)
        if "resource_already_exists_exception" in detail:
            logger.info(f"Index '{index}' was created concurrently, continuing")
            return False
        # the two failures worth naming: both surface as a generic 400 and
        # neither is fixable from this service
        if INGEST_PIPELINE in detail:
            logger.error(
                f"'{index}' could not be created : the ingest pipeline "
                f"'{INGEST_PIPELINE}' does not exist on this cluster. It is what "
                f"fills text_embedding, so it has to be deployed first."
            )
        elif SYNONYMS_SET in detail:
            logger.error(
                f"'{index}' could not be created : the synonyms set "
                f"'{SYNONYMS_SET}' does not exist on this cluster."
            )
        logger.exception(f"Could not create index '{index}': {type(e).__name__}: {e}")
        raise


def indexed_fids(fids, index=None, client=None):
    """
    Which of these fIds already have chunks in the index.

    This is what makes a second exam on the same document cheap. Ingestion is
    download plus OCR plus embedding -- seconds to minutes per file -- and
    repeating it for a document already there would be the slowest part of a
    request that needed to do no work at all.

    An error is not fatal: reporting nothing as indexed re-ingests, which is
    wasteful but correct, where reporting everything as indexed would silently
    build a paper from documents that were never read.
    """
    # stripped, not just tested for truth : "  " is a truthy string that matches
    # nothing, so leaving it in costs a query and returns a misleading answer
    fids = [str(fid).strip() for fid in (fids or [])]
    fids = [fid for fid in fids if fid]
    if not fids:
        return set()

    index = index or EXAM_INDEX
    client = client or get_es_client()
    try:
        response = client.search(
            index=index,
            body={
                "size": 0,
                "query": {"terms": {"fId": fids}},
                "aggs": {"fids": {"terms": {"field": "fId", "size": len(fids)}}},
            },
        )
        buckets = response["aggregations"]["fids"]["buckets"]
        return {bucket["key"] for bucket in buckets if bucket.get("doc_count")}
    except Exception as e:
        # a missing index is the normal first-run case, not a problem
        if "index_not_found_exception" in str(e):
            return set()
        logger.warning(
            f"Could not check which fIds are in '{index}' ({e}), treating them all as new"
        )
        return set()


def delete_fid(fid, index=None, client=None):
    """
    Removes every chunk of one document, so a re-ingestion replaces rather than
    doubles it.

    Chunk ids are deterministic, so re-running an identical file would overwrite
    its own chunks anyway. This is for the case that is not identical -- the file
    was re-uploaded with more pages, or the chunk size changed -- where the old
    chunks have ids the new run never writes and would otherwise stay behind and
    keep being retrieved.
    """
    index = index or EXAM_INDEX
    client = client or get_es_client()
    try:
        response = client.delete_by_query(
            index=index,
            body={"query": {"term": {"fId": fid}}},
            refresh=True,
            conflicts="proceed",
        )
        deleted = response.get("deleted", 0)
        if deleted:
            logger.info(f"Removed {deleted} existing chunk(s) for fId={fid} from '{index}'")
        return deleted
    except Exception as e:
        if "index_not_found_exception" in str(e):
            return 0
        logger.warning(f"Could not clear existing chunks for fId={fid}: {e}")
        return 0


def bulk_index(documents, index=None, client=None, refresh=True):
    """
    Writes chunks and, by default, waits for them to become searchable.

    documents carry at least fId, text and pageNo. Each is routed by its fId,
    which the mapping requires and which keeps one document's chunks together.

    The refresh is the point of doing this synchronously. Elasticsearch makes a
    write searchable on its own schedule -- a second by default -- and a caller
    that retrieves the instant this returns would otherwise search the index as
    it was before the ingestion and get nothing, which surfaces to the user as
    "no matching documents found" for a file just read.
    """
    if not documents:
        return 0, []

    index = index or EXAM_INDEX
    client = client or get_es_client()

    from elasticsearch.helpers import bulk

    actions = []
    for document in documents:
        fid = document["fId"]
        actions.append({
            "_index": index,
            # deterministic, so re-running an identical ingestion overwrites its
            # own chunks instead of duplicating the document
            "_id": f"{fid}_{document.get('pageNo', 0)}_{document.get('para', 0)}",
            "_routing": fid,
            "_source": document,
        })

    start = time.time()
    written, errors = bulk(
        client,
        actions,
        chunk_size=BULK_CHUNK,
        # collect every failure rather than raising on the first, so one bad
        # chunk does not lose the rest of the document
        raise_on_error=False,
        request_timeout=ES_REQUEST_TIMEOUT,
    )

    if errors:
        logger.error(
            f"{len(errors)} of {len(actions)} chunk(s) rejected by '{index}'. "
            f"First: {errors[0]}"
        )

    if refresh:
        try:
            client.indices.refresh(index=index)
        except Exception as e:
            # the chunks are written, just not searchable yet. Saying so beats
            # failing, because the wait is usually about a second
            logger.warning(
                f"Could not refresh '{index}' after writing ({e}) : the chunks are "
                f"stored but may not be searchable for another moment"
            )

    logger.info(
        f"Indexed {written}/{len(actions)} chunk(s) into '{index}' in {time.time() - start:.2f}s"
    )
    return written, errors


# ----------------------------------------------------------------------------
# the whole job
# ----------------------------------------------------------------------------

def build_documents(chunks, fid, file_name="", username="", path="", tenant_id=""):
    """
    Turns extracted chunks into indexable documents.

    Only fields index_properties() declares are written. An undeclared field
    would not fail -- dynamic mapping would take it -- it would just put a field
    in this index that teamsync_v1 does not have, on an index whose whole
    purpose is to behave like teamsync_v1.

    in_trash is written explicitly rather than left to default. The retrieval
    filter is {"term": {"in_trash": False}}, and a missing field does not match
    a term query, so a document without it is indexed, is correct, and is never
    found -- the worst of the three possible outcomes.

    text_embedding is not set here. The index's default_pipeline fills it in
    during the write, from the `text` field below.
    """
    documents = []
    for chunk in chunks:
        documents.append({
            "fId": fid,
            "text": chunk["text"],
            "pageNo": chunk.get("pageNo", 1),
            "para": chunk.get("para", 0),
            "fileName": file_name or fid,
            "in_trash": False,
            "username": username or "",
            "path": path or "",
            "tenantId": tenant_id or "",
        })
    return documents


def ingest_one(fid, miniourl, index=None, username="", path="", file_name="",
               force=False, es=None, minio=None, tenant_id=""):
    """
    Downloads, extracts, chunks and indexes one document.

    Returns a report rather than raising, so one unreadable file in a batch does
    not cost the caller the others -- the report says which failed and why, and
    the caller decides whether a paper can still be built.

    file_name is only a fallback. minio is asked what the upload actually
    recorded and that wins, because a fileName sent in can be stale while
    the object's own metadata cannot.
    """
    index = index or EXAM_INDEX
    report = {"fid": fid, "miniourl": miniourl, "status": "failed",
              "chunks": 0, "fileName": file_name, "reason": None}
    start = time.time()

    try:
        if not force:
            if fid in indexed_fids([fid], index=index, client=es):
                report.update(status="already_indexed", reason="already in the index")
                logger.info(f"fId={fid} is already indexed in '{index}', skipping")
                return report

        logger.info(f"Downloading fId={fid} from minio : {miniourl}")
        data, stored_name = download(miniourl, client=minio)
        # with no name recorded in minio, download answers with the object's
        # own name, usually the document id : the name sent in beats that
        if not file_name or stored_name != miniourl.rstrip("/").split("/")[-1]:
            file_name = stored_name
        chunks = extract_chunks(data, file_name=file_name)
        report["fileName"] = file_name

        if not chunks:
            # a real outcome, not an error: an image-only pdf with OCR
            # unavailable, or a file that genuinely has no text on it. Which
            # one is said, because the fixes are different -- install
            # tesseract, or send a different file
            reason = "no text could be extracted"
            if _ocr_failed:
                reason += " (it may be a scan : OCR is not available on this server)"
            report.update(status="empty", reason=reason)
            logger.warning(
                f"fId={fid} ({file_name!r}) produced no chunks : nothing to index"
            )
            return report

        # a re-ingestion of a file whose page count or chunking changed would
        # otherwise leave the old chunks behind, with ids this run never writes
        if force:
            delete_fid(fid, index=index, client=es)

        documents = build_documents(
            chunks, fid, file_name=file_name, username=username, path=path,
            tenant_id=tenant_id,
        )
        logger.info(f"Indexing {len(documents)} chunk(s) of fId={fid} into '{index}'")
        written, errors = bulk_index(documents, index=index, client=es)

        report["chunks"] = written
        if errors and not written:
            report.update(status="failed", reason=f"every chunk was rejected: {errors[0]}")
        elif errors:
            report.update(status="partial", reason=f"{len(errors)} chunk(s) rejected")
        else:
            report["status"] = "indexed"

        logger.info(
            f"Ingested fId={fid} ({file_name!r}) : {written} chunk(s) into '{index}' "
            f"in {time.time() - start:.2f}s"
        )
        return report
    except Exception as e:
        logger.exception(
            f"Ingestion failed for fId={fid} ({miniourl}) after "
            f"{time.time() - start:.2f}s : {type(e).__name__}: {e}"
        )
        report["reason"] = f"{type(e).__name__}: {e}"
        return report


def normalise(document):
    """
    Accepts a document descriptor, or anything close enough to one, and returns
    just the fields ingestion needs.

    The field names are the ones the team sync new-document message already
    uses, and are deliberately not pythonic -- fileName is camelCase, miniourl is
    spelled the way it arrives -- so a caller holding one can pass it straight
    through. The handful of aliases below are for callers that never had such a
    message, not an invitation to rename anything.
    """
    if not isinstance(document, dict):
        raise ValueError(
            f"each document must be a dict, got {type(document).__name__}"
        )

    def pick(*names):
        for name in names:
            value = document.get(name)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    fid = pick("fid", "fId", "fileId", "file_id")
    url = pick("miniourl", "minioUrl", "minio_url", "url")

    tenant_id = pick("tenant_id", "tenantId")

    if not fid:
        raise ValueError(f"a document has no fid: {sorted(document)}")
    if not url:
        raise ValueError(f"document {fid} has no miniourl: {sorted(document)}")
    # every chunk records it and every published result carries it, so a
    # document arriving without one is refused rather than written blank
    if not tenant_id:
        raise ValueError(f"document {fid} has no tenant_id: {sorted(document)}")

    return {
        "fid": fid,
        "miniourl": url,
        # everything below is optional. fileName is only a fallback: download()
        # asks minio for the name the upload actually recorded, and prefers that.
        "fileName": pick("fileName", "filename", "file_name"),
        "username": pick("username", "userName", "user"),
        "path": pick("path"),
        "tenant_id": tenant_id,
        # a document may name the index it belongs in. Letting it carry that is
        # what makes a forwarding caller a one liner : ingest(doc) routes itself.
        "index": pick("index"),
    }


def ingest(documents, index=None, force=False):
    """
    THE function. Everything this module does, behind one call.

    Give it a document -- one, or a list of them -- and it downloads each
    document from minio, extracts the text (OCR for pages with no text layer),
    cuts it into overlapping chunks, and indexes them into elasticsearch, which
    fills in the embedding through the index's default pipeline.

        from doc_ingest import ingest

        report = ingest(document)             # one document
        report = ingest(documents)            # a batch of them
        report = ingest(document, index="x")  # override where it lands

    A document may carry its own "index", so a caller that just forwards what
    it received needs nothing else. The `index` argument wins when given, and
    EXAM_TARGET_INDEX is the fallback when neither says.

    fid, miniourl and tenant_id are required on each document. tenant_id is
    written onto every chunk as tenantId, the name teamsync_v1 uses for it.
    fileName, username and path are recorded on the chunks when present.

    Returns, and does not raise, for anything one document can do wrong:

        {"index": str, "requested": int, "indexed": int, "skipped": int,
         "failed": int, "chunks": int, "documents": [one report each]}

    Per document, "status" is one of:

        indexed          written and searchable
        partial          written, but elasticsearch rejected some chunks
        already_indexed  it was there already, nothing was downloaded
        empty            it was read and had no text in it
        failed           "reason" says what went wrong

    It raises only for things that make the whole call meaningless: a document
    with no fid, no url or no tenant_id, and an index that cannot be created.
    """
    if isinstance(documents, dict):
        documents = [documents]
    documents = [normalise(document) for document in (documents or [])]

    # resolved per document so a batch spanning two indices still works, though
    # a caller handling one document at a time is the normal case
    for document in documents:
        document["index"] = index or document["index"] or EXAM_INDEX

    default_index = index or (documents[0]["index"] if documents else EXAM_INDEX)
    if not documents:
        return {"index": default_index, "requested": 0, "indexed": 0, "skipped": 0,
                "failed": 0, "chunks": 0, "documents": []}

    start = time.time()

    # each distinct index created once for the batch, not once per document
    for target in dict.fromkeys(document["index"] for document in documents):
        try:
            ensure_index(target)
        except Exception as e:
            # nothing can be written, so every document would fail for the same
            # reason and saying it once is clearer than saying it per file
            raise RuntimeError(f"the index '{target}' is not usable: {e}")

    es = get_es_client()
    minio = get_minio_client()

    # everything already present is found in one query per index rather than one
    # per file -- this is what makes a repeat request cheap. The tenant plays no
    # part : a fid is either in the index or it is not, and nothing filters
    # retrieval on the tenant either.
    present = set()
    if not force:
        for target in dict.fromkeys(document["index"] for document in documents):
            fids = [d["fid"] for d in documents if d["index"] == target]
            present |= {(target, fid) for fid in indexed_fids(fids, index=target, client=es)}
        if present:
            logger.info(f"{len(present)} of {len(documents)} document(s) are already indexed")

    reports = []
    for document in documents:
        if (document["index"], document["fid"]) in present:
            reports.append({"fid": document["fid"], "miniourl": document["miniourl"],
                            "status": "already_indexed", "chunks": 0,
                            "fileName": document["fileName"],
                            "reason": "already in the index"})
            continue
        reports.append(ingest_one(
            document["fid"], document["miniourl"], index=document["index"],
            username=document["username"], path=document["path"],
            file_name=document["fileName"], force=force, es=es, minio=minio,
            tenant_id=document["tenant_id"],
        ))

    summary = {
        "index": default_index,
        "requested": len(documents),
        "indexed": sum(1 for r in reports if r["status"] in ("indexed", "partial")),
        "skipped": sum(1 for r in reports if r["status"] in ("already_indexed", "empty")),
        "failed": sum(1 for r in reports if r["status"] == "failed"),
        "chunks": sum(r["chunks"] for r in reports),
        "documents": reports,
    }
    logger.info(
        f"Ingestion finished in {time.time() - start:.2f}s : {summary['indexed']} indexed, "
        f"{summary['skipped']} skipped, {summary['failed']} failed, "
        f"{summary['chunks']} chunk(s)"
    )
    return summary


# ----------------------------------------------------------------------------
# script entry point
# ----------------------------------------------------------------------------

def reload_config(env_file=None):
    """
    Re-reads every setting in SETTINGS from the environment.

    The module constants are bound at import, which is right for a service --
    exam_main loads its .env before importing anything, so the values are there
    already. A script is the other way round: __main__ runs after the import, so
    without this the CLI would run against the defaults and quietly talk to the
    wrong cluster.
    """
    if env_file:
        try:
            from dotenv import load_dotenv
            # override=False so a real environment variable still wins, matching
            # how exam_main loads the same file
            load_dotenv(env_file, override=False)
        except Exception as e:
            logger.warning(f"Could not load {env_file}: {e}")

    _apply_settings()
    # a client built against the old settings would still point at the old host
    module = sys.modules[__name__]
    module._minio_client = None
    module._es_client = None


def main(argv=None):
    """
    Runs one ingestion from the command line, which is the quickest way to
    backfill a document or to see what a file that produced a bad paper actually
    extracted to.
    """
    parser = argparse.ArgumentParser(
        description="Download documents from MinIO, extract their text, and index "
                    "overlapping chunks into Elasticsearch."
    )
    parser.add_argument("--fid", action="append", required=True,
                        help="document id, repeat once per document")
    parser.add_argument("--url", action="append", required=True,
                        help="matching bucket qualified minio key, repeat once per document")
    parser.add_argument("--index", default=EXAM_INDEX, help=f"target index (default {EXAM_INDEX})")
    parser.add_argument("--tenant-id", default="",
                        help="tenant to record on each chunk; required unless --dry-run")
    parser.add_argument("--username", default="", help="username to record on each chunk")
    parser.add_argument("--path", default="", help="path to record on each chunk")
    parser.add_argument("--force", action="store_true",
                        help="re-ingest documents already in the index")
    parser.add_argument("--dry-run", action="store_true",
                        help="download and extract, but write nothing")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--env-file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        help="settings to load before connecting (default: the .env beside this "
             "file). Real environment variables still win.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.strip().upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # the module constants were bound at import, which happened before this file
    # was read -- without this the script runs against the defaults
    explicit_index = args.index != EXAM_INDEX
    reload_config(args.env_file if os.path.exists(args.env_file) else None)
    if not explicit_index:
        args.index = EXAM_INDEX
    logger.info(
        f"doc_ingest : index={args.index} es={ES_HOST}:{ES_PORT} minio={MINIO_ENDPOINT}"
    )

    if len(args.fid) != len(args.url):
        parser.error(f"got {len(args.fid)} --fid and {len(args.url)} --url, they must pair up")
    if not args.dry_run and not args.tenant_id.strip():
        parser.error("--tenant-id is required : every chunk records the tenant it was for")

    if args.dry_run:
        summary = {"index": args.index, "dry_run": True, "documents": []}
        for fid, url in zip(args.fid, args.url):
            try:
                data, file_name = download(url)
                chunks = extract_chunks(data, file_name=file_name)
                summary["documents"].append({
                    "fid": fid, "fileName": file_name, "chunks": len(chunks),
                    "first_chunk": chunks[0]["text"][:300] if chunks else "",
                })
            except Exception as e:
                summary["documents"].append({"fid": fid, "error": f"{type(e).__name__}: {e}"})
    else:
        summary = ingest(
            [{"fid": fid, "miniourl": url, "tenant_id": args.tenant_id,
              "username": args.username, "path": args.path}
             for fid, url in zip(args.fid, args.url)],
            index=args.index, force=args.force,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
