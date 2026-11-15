import unittest

from evals.common import is_relevant, matches
from rag.answer import _validate, pack_context
from rag.ingest import Doc, chunk_fixed, chunk_structure, content_hash, find_near_duplicates, n_tokens, normalize
from rag.search import Filters, tokenize


def doc(doc_id: str, text: str) -> Doc:
    return Doc(doc_id, doc_id, "", "open_api", "html", "2026-01-01", text, content_hash(text), None, [])


class Dedup(unittest.TestCase):
    def test_hash_ignores_case_and_whitespace(self):
        self.assertEqual(content_hash("Rate  limits\nare shared"), content_hash("rate limits are   shared"))

    def test_normalize_strips_invisible_characters(self):
        self.assertEqual(normalize("a\u200bb\u00a0c\r\n\n\n\nd"), "ab c\n\nd")

    def test_near_duplicates_flagged_but_distinct_docs_not(self):
        base = " ".join(f"word{i}" for i in range(200))
        docs = [doc("a", base), doc("b", base + " extra tail words here"), doc("c", " ".join(f"other{i}" for i in range(200)))]
        pairs = find_near_duplicates(docs, 0.5)
        self.assertEqual([(a, b) for a, b, _ in pairs], [("a", "b")])


class Chunking(unittest.TestCase):
    TEXT = (
        "# Title\n\nIntro paragraph that is long enough to stand alone. " * 3
        + "\n\n## Setup\n\n"
        + "\n\n".join(f"Paragraph {i} " + "with some words " * 30 for i in range(6))
        + "\n\n## Example\n\n```\n" + "\n".join(f'"field{i}": {i},' for i in range(40)) + "\n```\n\n"
        + "## Limits\n\n| Limit | Value |\n|---|---|\n" + "\n".join(f"| per unit {i} | {i} |" for i in range(10))
    )

    def test_structure_respects_budget_and_keeps_blocks_whole(self):
        chunks = chunk_structure(self.TEXT, max_tokens=200, min_tokens=30)
        for c in chunks:
            self.assertLessEqual(n_tokens(c["text"]), 200 + 30)
            self.assertEqual(c["text"].count("```") % 2, 0, "code fence split across chunks")
        table = [c for c in chunks if "| Limit | Value |" in c["text"]]
        self.assertTrue(table and all(c["section_path"][-1] == "Limits" for c in table))

    def test_fixed_windows_overlap(self):
        chunks = chunk_fixed(self.TEXT, size=100, overlap=20)
        self.assertTrue(all(n_tokens(c["text"]) <= 101 for c in chunks))
        self.assertGreater(len(chunks), 3)


class Search(unittest.TestCase):
    def test_tokenizer_keeps_identifiers_and_parts(self):
        toks = tokenize("Use reservation.created.v2 and GET /reservations-v3")
        self.assertIn("reservation.created.v2", toks)
        self.assertIn("created", toks)
        self.assertIn("reservations-v3", toks)

    def test_filters(self):
        c = {"doc_set": "open_api", "updated_at": "2026-05-01", "format": "pdf"}
        self.assertTrue(Filters(doc_set="open_api", updated_after="2026-01-01", formats=("pdf",)).allows(c))
        self.assertFalse(Filters(doc_set="booking_engine").allows(c))
        self.assertFalse(Filters(updated_after="2026-06-01").allows(c))


class Evals(unittest.TestCase):
    def test_relevance_needs_doc_and_quote(self):
        item = {"evidence": [{"doc_id": "d1", "quote": "five access   tokens"}]}
        self.assertTrue(is_relevant({"doc_id": "d1", "text": "a maximum of Five access tokens per day"}, item))
        self.assertFalse(is_relevant({"doc_id": "d2", "text": "a maximum of five access tokens per day"}, item))

    def test_key_fact_alternatives(self):
        self.assertTrue(matches("Retires on March 1, 2027.", "1 March 2027|March 1, 2027"))
        self.assertFalse(matches("Retires soon.", "1 March 2027|March 1, 2027"))


class Answering(unittest.TestCase):
    def test_context_budget(self):
        chunk = {"title": "T", "section": "S", "doc_set": "open_api", "updated_at": "2026-01-01", "text": "word " * 300}
        sources = pack_context([(chunk, 1.0)] * 5, budget=700)
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(pack_context([(chunk, 1.0)], budget=100)), 1, "first chunk is truncated, not dropped")

    def test_validation_rejects_unknown_or_missing_citations(self):
        ok = '{"status": "answered", "claims": [{"text": "x", "sources": ["S1"]}], "missing": ""}'
        self.assertIsNotNone(_validate(ok, {"S1"})[0])
        self.assertIn("unknown sources", _validate(ok.replace("S1", "S9"), {"S1"})[1])
        self.assertIsNotNone(_validate('{"status": "answered", "claims": [{"text": "x", "sources": []}], "missing": ""}', {"S1"})[1])
        self.assertIsNotNone(_validate('{"status": "answered", "claims": [], "missing": ""}', {"S1"})[1])


if __name__ == "__main__":
    unittest.main()
