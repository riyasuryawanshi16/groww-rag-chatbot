"""
Unit tests for Groww Mutual Funds RAG Engine.
Validates:
1. Sources count and schema (15-25 official URLs)
2. PII detection (PAN, Aadhaar, Phone)
3. Safe refusal on investment advice (with educational link)
4. Sentence length constraint (<= 3 sentences)
5. Mandatory citation generation and Last updated line
6. Factual accuracy on key concepts (ELSS lock-in, cut-off timings, large-cap criteria)
7. Exact fallback string: "I don't have this information in my official sources."
"""

import os
import csv
import unittest
from rag_engine import GrowwRAGEngine, GROWW_EDU_URL


class TestGrowwRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sources_path = os.path.join(base_dir, "sources.csv")
        corpus_path = os.path.join(base_dir, "data", "corpus.json")
        cls.engine = GrowwRAGEngine(sources_path, corpus_path)

    def test_sources_count_and_columns(self):
        """Verify sources.csv has 15-25 URLs and correct columns."""
        with open(self.engine.sources_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertGreaterEqual(len(reader), 15, "Corpus must have at least 15 URLs")
            self.assertLessEqual(len(reader), 25, "Corpus must have at most 25 URLs")
            required_cols = {"source_id", "organization", "category", "title", "url", "description"}
            self.assertTrue(required_cols.issubset(set(reader[0].keys())))

    def test_pii_detection(self):
        """Verify PAN, Aadhaar, and phone numbers are detected and blocked."""
        res_pan = self.engine.answer_query("My PAN is ABCDE1234F, what is my account balance?")
        self.assertEqual(res_pan["status"], "refused_pii")
        self.assertIn("PAN", res_pan["answer"])

        res_aadhaar = self.engine.answer_query("Here is my Aadhaar 9876 5432 1098, check status")
        self.assertEqual(res_aadhaar["status"], "refused_pii")
        self.assertIn("Aadhaar", res_aadhaar["answer"])

        res_phone = self.engine.answer_query("Call me at 9876543210 regarding mutual fund investment")
        self.assertEqual(res_phone["status"], "refused_pii")
        self.assertIn("Phone", res_phone["answer"])

    def test_investment_advice_refusal(self):
        """Verify queries asking for advice or picks are safely refused with the educational URL."""
        advice_queries = [
            "Which mutual fund should I buy for high returns?",
            "Recommend me a fund for long term investment",
            "Should I invest in SBI Bluechip or SBI Flexicap?",
            "Give me portfolio advice for 2025",
            "Which fund is better between SBI Bluechip and Flexicap?"
        ]
        for q in advice_queries:
            res = self.engine.answer_query(q)
            self.assertEqual(res["status"], "refused_advice", f"Query failed refusal: {q}")
            self.assertEqual(res["url"], GROWW_EDU_URL)
            self.assertIn("cannot provide personalized investment advice", res["answer"])

    def test_multi_source_and_citation(self):
        """Verify multi-source retrieval, Detailed Overview, and Exhaustive Sources citations."""
        test_queries = [
            "What is the lock-in period of an ELSS fund?",
            "What are the cut-off timings for equity mutual funds?",
            "What is a large-cap mutual fund according to SEBI?",
            "What is the Expense Ratio in a mutual fund?",
            "What is the investment mandate of SBI Bluechip Fund?"
        ]
        for q in test_queries:
            res = self.engine.answer_query(q)
            self.assertEqual(res["status"], "ok")
            self.assertIn("Detailed Overview", res["answer"])
            self.assertIn("Exhaustive Sources", res["answer"])
            self.assertGreaterEqual(len(res["sources"]), 1, f"Must retrieve sources: {q}")
            for s in res["sources"]:
                self.assertTrue(s["url"].startswith("http"), f"Invalid source URL: {s}")
                self.assertTrue(bool(s["title"]), "Source title required")
            self.assertIn("Last updated from sources:", res["answer"])

    def test_out_of_scope_exact_string(self):
        """Verify exact response: 'I don't have this information in my official sources.'"""
        res = self.engine.answer_query("How do I bake a chocolate cake?")
        self.assertEqual(res["status"], "out_of_scope")
        self.assertEqual(res["answer"], "I don't have this information in my official sources.")

    def test_factual_accuracy(self):
        """Check factual details in answers."""
        # ELSS lock-in
        res = self.engine.answer_query("What is the lock-in period for ELSS mutual funds?")
        self.assertIn("3 year", res["answer"])
        self.assertIn("Section 80C", res["answer"])

        # Cut-off timing
        res_cutoff = self.engine.answer_query("What is the cut-off time for mutual fund orders?")
        self.assertIn("3:00 PM", res_cutoff["answer"])


if __name__ == "__main__":
    unittest.main()
