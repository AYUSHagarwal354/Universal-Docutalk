"""
    this reranks the chunks returned by elasticsearch
    - RRF only merges the positions of chunks in the BM25 and kNN lists, it never reads the text
    - the reranker (cross-encoder on watsonx.ai) reads the question and each chunk together
      and scores how well the chunk answers the question
    - only the chunks that pass the score checks are sent to the model and shown as references
    - chunks too short to hold an answer and chunks that repeat an earlier one are dropped before reranking
"""
import os
import math
import logging
import requests
from query_enhancement import get_token

logger = logging.getLogger(__name__)

PROJECT_ID_IBM = os.getenv("PROJECT_ID_IBM")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_URL = os.getenv("RERANK_URL", "https://us-south.ml.cloud.ibm.com/ml/v1/text/rerank?version=2023-10-25")
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cross-encoder/ms-marco-minilm-l-12-v2")
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", 10))
RERANK_TRUNCATE_TOKENS = int(os.getenv("RERANK_TRUNCATE_TOKENS", 512))

RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))   # elasticsearch hits sent to the reranker
RERANK_KEEP = int(os.getenv("RERANK_KEEP", 10))               # max chunks kept
RERANK_MIN_KEEP = int(os.getenv("RERANK_MIN_KEEP", 5))        # best chunks sent to the model and shown as references when none passes the floor (0 = send nothing)
# raw model score of ms-marco-minilm-l-12-v2, measured on real questions: a chunk that answers the question
# scores above 0, a chunk that is only related scores about -3 to -6, and for a question the documents
# cannot answer even the best chunk stays below -9
RERANK_SCORE_FLOOR = float(os.getenv("RERANK_SCORE_FLOOR", -8.0))    # chunks below this are unrelated and never kept
# chunks shorter than this (after removing [CLS]/[SEP]) are single words, "Top of Form", page headers ...
RERANK_MIN_CHUNK_CHARS = int(os.getenv("RERANK_MIN_CHUNK_CHARS", 20))


def rerank_hits(query, hits):
    """
    returns (hits, relevant): the hits to send to the model, best first, with _score replaced by the reranker score (0-1).
    relevant is False when no hit passed the floor and the hits are only the best RERANK_MIN_KEEP as a fallback.
    if the reranker is off or fails, the elasticsearch order is kept (no score checks).
    """
    candidates = clean_hits(hits)[:RERANK_CANDIDATES]
    if not candidates:
        return [], False

    scores = score_texts(query, [hit["_source"].get("text", "") for hit in candidates]) if RERANK_ENABLED else None
    if scores is None:
        logger.warning("Reranker not used --> keeping elasticsearch order")
        return candidates[:RERANK_KEEP], True

    for hit, score in zip(candidates, scores):
        hit["_score"] = score
    ranked = sorted(candidates, key=lambda hit: hit["_score"], reverse=True)

    # the best chunks above the floor are kept, however far they are below the best one
    kept = [hit for hit in ranked if hit["_score"] >= RERANK_SCORE_FLOOR][:RERANK_KEEP]
    relevant = bool(kept)

    logger.info(f"Rerank: kept {len(kept)}/{len(candidates)} chunks (floor={RERANK_SCORE_FLOOR}) scores={[(round(hit['_score'], 4), hit['_source'].get('fileName', '')) for hit in ranked]}")

    # even the best chunk is below the floor: the best chunks are still sent so the model can try to answer
    if not relevant:
        kept = ranked[:RERANK_MIN_KEEP]
        logger.warning(f"Rerank: no chunk passed the floor --> sending the best {len(kept)} to the model")

    # the floor uses the raw model score, the response carries it as a 0-1 score
    for hit in kept:
        hit["_score"] = 1 / (1 + math.exp(-hit["_score"]))
    return kept, relevant


def clean_hits(hits):
    """drops chunks too short to hold an answer and chunks whose text repeats an earlier chunk (copies of a file, boilerplate)"""
    kept, seen = [], set()
    for hit in hits:
        text = " ".join(hit["_source"].get("text", "").replace("[CLS]", " ").replace("[SEP]", " ").split()).lower()
        if len(text) < RERANK_MIN_CHUNK_CHARS or text in seen:
            continue
        seen.add(text)
        kept.append(hit)
    if len(kept) < len(hits):
        logger.info(f"Clean: dropped {len(hits) - len(kept)}/{len(hits)} short or repeated chunks")
    return kept


def score_texts(query, texts):
    """calls the watsonx.ai rerank API and returns the raw model score per text (same order), or None if the call fails"""
    body = {
        "model_id": RERANK_MODEL_ID,
        "project_id": PROJECT_ID_IBM,
        "query": query,
        "inputs": [{"text": text} for text in texts],
        "parameters": {"truncate_input_tokens": RERANK_TRUNCATE_TOKENS},
    }
    try:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": get_token(),
        }
        response = requests.post(RERANK_URL, headers=headers, json=body, timeout=RERANK_TIMEOUT)
        response.raise_for_status()
        scores = [float("-inf")] * len(texts)   # a text missing from the response counts as unrelated
        for result in response.json()["results"]:
            scores[result["index"]] = result["score"]
        return scores
    except Exception as e:
        logger.error(f"Rerank call failed: {e}")
        return None
