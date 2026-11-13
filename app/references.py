"""
    picks the reference files from the answer itself
    - the prompts make the model name the files it used inside <font> tags
      ("Sources: ..." at the end of the answer, or "Source: ..." in the <EXPLANATION> of the thinking prompts)
    - after the answer is streamed, only the files it names, out of the files it was given, are sent as meta_data
"""
import re
import html
import logging

logger = logging.getLogger(__name__)

FONT_PATTERN = re.compile(r"<font[^>]*>(.*?)</font>", re.IGNORECASE | re.DOTALL)
SOURCES_PATTERN = re.compile(r"sources?\s*:", re.IGNORECASE)
# the fallback message of the prompts, with a straight or a curly apostrophe
NOT_FOUND_PATTERN = re.compile(r"couldn['’]t find information on this topic", re.IGNORECASE)
# an answer whose opening sentence says the documents do not cover the question ("The reference text does not mention ...")
NOT_COVERED_PATTERN = re.compile(
    r"\b(?:does not|do not|doesn['’]t|don['’]t|did not)\s+(?:contain|mention|provide|include|discuss|cover|have)\b"
    r"|\bno (?:information|mention)\b|\bnot (?:mentioned|found|available|provided)\b", re.IGNORECASE)


def _strip_tags(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def cited_names(answer):
    """file names the answer cites: the text of its <font> tags, or else the comma separated list after its last "Sources:" """
    names = [_strip_tags(name).strip() for name in FONT_PATTERN.findall(answer)]
    if not names:
        parts = SOURCES_PATTERN.split(answer)
        if len(parts) > 1:
            lines = _strip_tags(re.sub(r"</p>|<br\s*/?>", "\n", parts[-1], flags=re.IGNORECASE)).strip().splitlines()
            names = [name.strip() for name in lines[0].split(",")] if lines else []
    return [name for name in names if name]


def _opening_sentence(answer):
    """the first sentence of the answer's first paragraph; headings are skipped"""
    paragraph = re.search(r"<p[^>]*>(.*?)</p>", answer, re.IGNORECASE | re.DOTALL)
    text = " ".join(re.sub(r"<[^>]+>", " ", html.unescape(paragraph.group(1) if paragraph else answer)).split())
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]


def _normalize(name):
    """case, the extension, underscores, hyphens and spaces do not matter when a cited name is matched to a file"""
    name = re.sub(r"\.[a-z0-9]{2,5}$", "", name.strip().lower())
    return re.sub(r"[\s_\-]+", " ", name).strip()


def used_references(answer, references, relevant=True):
    """
    the references (meta_data entries) of the files the answer cites, in the order they are cited.
    names that are not among the given files are ignored; a "not found" answer gets no references.
    when nothing passed the reranker (relevant False), an answer whose opening sentence says the documents do not
    cover the question gets no references either: the model sometimes still lists the files it was given.
    a partial answer ("X is ... . Further details are not provided.") keeps its references.
    """
    if NOT_FOUND_PATTERN.search(answer):
        logger.info("The answer is the fallback message --> no references")
        return []
    if not relevant and NOT_COVERED_PATTERN.search(_opening_sentence(answer)):
        logger.info("Nothing passed the reranker and the answer says the documents do not cover the question --> no references")
        return []
    by_name = {}
    for reference in references:
        by_name.setdefault(_normalize(reference["filename"]), []).append(reference)

    names = cited_names(answer)
    used, seen = [], set()
    for name in names:
        # one <font> tag can hold several names: "a.pdf, b.pdf"
        matches = by_name.get(_normalize(name)) or [ref for part in name.split(",") for ref in by_name.get(_normalize(part), [])]
        for reference in matches:
            if reference["fId"] not in seen:
                seen.add(reference["fId"])
                used.append(reference)

    if not names:
        logger.warning("The answer cites no source --> no references")
    else:
        logger.info(f"Sources cited in the answer {names} --> references {[ref['filename'] for ref in used]}")
    return used
