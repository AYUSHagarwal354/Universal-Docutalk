"""
Tests for references.py

Covers:
  cited_names:
    - the Sources line of the NLP prompt (<font> tags, any colour)
    - the "Source:" lines of the thinking prompts' <EXPLANATION>
    - several names in one <font> tag, HTML entities
    - a plain "Sources: a, b" line when there is no <font> tag
    - no sources → []

  used_references:
    - only cited files that were given to the model, in the order they are cited, once per file
    - case, extension, underscores, hyphens and spaces do not matter
    - two files with the same name are both returned
    - the fallback message (straight or curly apostrophe) or no cited name → []
    - nothing passed the reranker and the answer's opening sentence says "the documents do not cover it" → [] even if it
      lists files; kept when the reranker found relevant chunks, or when the answer opens with real content
      (a later "further details are not provided" does not count)
"""

from references import cited_names, used_references


def _font(name, color="#6366F1"):
    return f'<font color="{color}">{name}</font>'


def _ref(fid, filename):
    return {"filename": filename, "fId": fid, "page_no": 1, "score": 0.9, "id": f"{fid}_1_0"}


REFS = [_ref("a", "Well_Completion_Report_GK-42-10_J.pdf"), _ref("b", "Finance_Policy_Handbook.pdf"),
        _ref("c", "Modern_Chinese_Maritime_Forces.pdf")]


class TestCitedNames:

    def test_reads_the_sources_line(self):
        answer = f"<h3>Answer</h3><p>text</p><p><strong>Sources:</strong> {_font('a.pdf')}, {_font('b.docx', 'Blue')}</p>"
        assert cited_names(answer) == ["a.pdf", "b.docx"]

    def test_reads_the_source_lines_of_the_explanation(self):
        answer = (f"<EXPLANATION><p><strong>Source:</strong> {_font('a.pdf')}</p><p>chunk 1</p>"
                  f"<p><strong>Source:</strong> {_font('b.pdf')}</p><p>chunk 2</p></EXPLANATION><h3>Final</h3>")
        assert cited_names(answer) == ["a.pdf", "b.pdf"]

    def test_reads_several_names_in_one_tag_and_entities(self):
        assert cited_names(f"<p>Sources: {_font('R&amp;D plan.pdf, b.pdf')}</p>") == ["R&D plan.pdf, b.pdf"]

    def test_reads_a_plain_sources_line(self):
        answer = "<p>The answer.</p><p><strong>Sources:</strong> a.pdf, b.pdf</p><p>Ask me more!</p>"
        assert cited_names(answer) == ["a.pdf", "b.pdf"]

    def test_no_sources(self):
        assert cited_names("<p>Just an answer.</p>") == []


class TestUsedReferences:

    def test_returns_only_cited_files_in_cited_order(self):
        answer = f"<p>...</p><p>Sources: {_font('Modern_Chinese_Maritime_Forces.pdf')}, {_font('Well_Completion_Report_GK-42-10_J.pdf')}</p>"
        assert [r["fId"] for r in used_references(answer, REFS)] == ["c", "a"]

    def test_names_are_matched_loosely(self):
        answer = f"<p>Sources: {_font('well completion report gk 42 10 j')}, {_font('FINANCE_POLICY_HANDBOOK')}</p>"
        assert [r["fId"] for r in used_references(answer, REFS)] == ["a", "b"]

    def test_several_names_in_one_tag(self):
        answer = f"<p>Sources: {_font('Finance_Policy_Handbook.pdf, Modern_Chinese_Maritime_Forces.pdf')}</p>"
        assert [r["fId"] for r in used_references(answer, REFS)] == ["b", "c"]

    def test_files_not_given_to_the_model_are_ignored(self):
        answer = f"<p>Sources: {_font('Some_Other_File.pdf')}, {_font('Finance_Policy_Handbook.pdf')}</p>"
        assert [r["fId"] for r in used_references(answer, REFS)] == ["b"]

    def test_a_file_cited_twice_is_returned_once(self):
        answer = (f"<EXPLANATION><p>Source: {_font('Finance_Policy_Handbook.pdf')}</p></EXPLANATION>"
                  f"<p>Sources: {_font('Finance_Policy_Handbook.pdf')}</p>")
        assert [r["fId"] for r in used_references(answer, REFS)] == ["b"]

    def test_two_files_with_the_same_name_are_both_returned(self):
        refs = [_ref("x1", "report.pdf"), _ref("x2", "report.pdf"), _ref("y", "other.pdf")]
        assert [r["fId"] for r in used_references(f"<p>Sources: {_font('report.pdf')}</p>", refs)] == ["x1", "x2"]

    def test_fallback_message_gets_no_references(self):
        for apostrophe in ("'", "’"):
            answer = (f"<p><em>Thank you for your search! I couldn{apostrophe}t find information on this topic in your "
                      f"uploaded documents.</em></p><p>Sources: {_font('Finance_Policy_Handbook.pdf')}</p>")
            assert used_references(answer, REFS) == []

    def test_answer_citing_nothing_gets_no_references(self):
        assert used_references("<p>An answer without sources.</p>", REFS) == []

    # real answers from the live run on teamsync_v5
    CAKE = ("<h3>Recipe for Chocolate Cake</h3><p>The reference text does not contain a recipe for chocolate cake. "
            "The provided documents appear to be related to well completion reports and maritime forces.</p>"
            f"<p><strong>Sources:</strong> {_font('Well_Completion_Report_GK-42-10_J.pdf')}, {_font('Modern_Chinese_Maritime_Forces.pdf')}</p>")
    GHOSH = ("<h3>Personnel Information</h3><p>According to the provided document, Debashish Ghosh is mentioned as the person "
             "who approved the Well Completion Report of GK-42-10.</p>"
             f"<p><strong>Sources:</strong> {_font('Well_Completion_Report_GK-42-10_J.pdf')}</p>")

    def test_nothing_relevant_and_answer_says_not_covered_gets_no_references(self):
        assert used_references(self.CAKE, REFS, relevant=False) == []

    def test_partial_answer_keeps_references_when_chunks_were_relevant(self):
        assert [r["fId"] for r in used_references(self.CAKE, REFS, relevant=True)] == ["a", "c"]

    def test_real_answer_from_fallback_chunks_keeps_its_reference(self):
        assert [r["fId"] for r in used_references(self.GHOSH, REFS, relevant=False)] == ["a"]

    def test_not_covered_wording_late_in_a_long_answer_is_ignored(self):
        answer = ("<p>" + "The MDT samples were collected at 916m and yielded gas. " * 6 +
                  "The analysis results are not mentioned.</p>" + f"<p>Sources: {_font('Well_Completion_Report_GK-42-10_J.pdf')}</p>")
        assert [r["fId"] for r in used_references(answer, REFS, relevant=False)] == ["a"]

    def test_short_partial_answer_from_fallback_chunks_keeps_its_reference(self):
        # the answer from the local run for "Who is d.gosh?": its second sentence says "not provided",
        # but it opens with the answer, so the cited file is a real reference
        answer = ("<h3>Author Identification</h3><p>D. Ghosh is identified as the GM(G)-HGOG, which suggests a managerial or "
                  "executive role, possibly in the oil and gas industry, given the context of the well completion report.</p>"
                  "<p>Further details about D. Ghosh's background, expertise, or specific responsibilities are not provided in the "
                  "available text.</p>"
                  f"<p><strong>Sources:</strong> {_font('Well_Completion_Report_GK-42-10_J.pdf', '#FF9933')}</p>")
        assert [r["fId"] for r in used_references(answer, REFS, relevant=False)] == ["a"]

    def test_not_covered_opening_after_a_heading_gets_no_references(self):
        answer = ("<h3>Capital of France</h3><p>The reference text does not mention the capital of France. It covers well "
                  f"reports.</p><p>Sources: {_font('Finance_Policy_Handbook.pdf')}</p>")
        assert used_references(answer, REFS, relevant=False) == []
