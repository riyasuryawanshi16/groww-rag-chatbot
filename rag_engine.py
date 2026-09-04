"""Exhaustive Financial Research & RAG Engine for Groww Mutual Funds.
Implements multi-source fallback retrieval, embedded default corpus with deep links, 
and comprehensive safety guardrails (PII, Investment Advice, and Toxicity/Abuse filters).
"""

import os
import re
import csv
import json
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GROWW_EDU_URL = "https://groww.in/blog"

# Comprehensive FAQ corpus featuring direct, deep-linked URLs to official Groww educational articles
DEFAULT_MUTUAL_FUND_CORPUS = [
    {
        "source_id": "faq_sip",
        "title": "What is a Systematic Investment Plan (SIP) in mutual funds?",
        "category": "SIP Basics",
        "organization": "Groww",
        "content": "A Systematic Investment Plan (SIP) is a method of investing a fixed sum regularly in mutual funds. It allows you to buy units on a set date each month, enforcing financial discipline. SIPs use rupee-cost averaging, meaning you buy more units when markets are low and fewer units when markets are high, reducing the impact of market volatility over the long term.",
        "url": "https://groww.in/mutual-funds/start-sip"
    },
    {
        "source_id": "faq_mf_basics",
        "title": "What is a mutual fund and how does it work?",
        "category": "Mutual Fund Basics",
        "organization": "Groww",
        "content": "A mutual fund is a professionally managed investment fund that pools money from multiple investors to purchase securities like stocks, bonds, and other assets. Managed by an Asset Management Company (AMC) and professional portfolio managers, it gives everyday investors access to diversified portfolios tailored to specific risk appetites and financial goals.",
        "url": "https://groww.in/mutual-funds"
    },
    {
        "source_id": "faq_1",
        "title": "What are the cut-off timings for mutual fund purchases to get same-day NAV?",
        "category": "Cut-off Timings",
        "organization": "Groww",
        "content": "For equity mutual funds and hybrid funds, the cut-off time is 3:00 PM. If your purchase order and fund realization happen before 3:00 PM on a working business day, you get the same day's NAV. For liquid and overnight funds, the cut-off time is typically earlier (around 1:30 PM). Orders placed after cut-off or on weekends/public holidays are processed on the next business day.",
        "url": "https://groww.in/help/mutual-funds/order/when-withdrawing-redeeming-which-days-nav-will-be-applicable-1"
    },
    {
        "source_id": "faq_2",
        "title": "What is the statutory lock-in period and tax benefit for ELSS funds?",
        "category": "ELSS Tax Saver",
        "organization": "Groww",
        "content": "Equity Linked Saving Schemes (ELSS) carry a mandatory 3-year lock-in period from the date of unit allotment, which is the shortest among all Section 80C tax-saving investment options. Investments up to Rs. 1.5 Lakh per financial year qualify for tax deductions under Section 80C of the Income Tax Act under the old tax regime.",
        "url": "https://groww.in/mutual-funds/category/best-elss-mutual-funds"
    },
    {
        "source_id": "faq_3",
        "title": "What is the difference between Direct and Regular mutual fund plans?",
        "category": "Direct vs Regular",
        "organization": "Groww",
        "content": "Direct mutual fund plans are purchased directly from the Asset Management Company (AMC) or through platforms like Groww without any intermediaries. Because there are no distributor commissions, Direct plans have a lower Total Expense Ratio (TER) compared to Regular plans, which translates into higher long-term returns for investors.",
        "url": "https://groww.in/mutual-funds"
    },
    {
        "source_id": "faq_4",
        "title": "How is exit load calculated on mutual fund redemptions?",
        "category": "Charges & Exit Load",
        "organization": "Groww",
        "content": "Exit load is a fee charged by mutual fund houses when an investor redeems their units within a specified duration (e.g., 1% if redeemed within 1 year). If you hold units beyond the exit load period specified in the scheme document, zero exit load is charged upon redemption.",
        "url": "https://groww.in/mutual-funds"
    },
    {
        "source_id": "faq_5",
        "title": "Large-Cap vs. Flexi-Cap mutual funds",
        "category": "Fund Categories",
        "organization": "Groww",
        "content": "Large-Cap mutual funds must invest at least 80% of their total assets in the top 100 companies by market capitalization as per SEBI regulations, offering stability and steady growth. Flexi-Cap funds have a dynamic mandate requiring a minimum of 65% investment in equities across large-cap, mid-cap, and small-cap stocks without market capitalization restrictions.",
        "url": "https://groww.in/mutual-funds"
    }
]

PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_REGEX = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

ADVICE_PATTERNS = [
    r"\bshould\s+i\s+(?:buy|sell|invest|choose|pick)\b",
    r"\bwhich\s+(?:fund|scheme|stock|plan)\s+(?:should\s+i|to\s+buy|to\s+invest|to\s+choose)\b",
    r"\bwhich\s+(?:fund|scheme|one)?\s*is\s+better\b",
    r"\b(?:better\s+between|better\s+to\s+invest)\b",
    r"\b(?:recommend|suggest)\s+(?:me\s+)?(?:a\s+)?(?:fund|scheme|portfolio|stock|investment)\b",
    r"\b(?:investment|portfolio)\s+(?:advice|recommendation|review|allocation)\b",
    r"\bwhere\s+(?:should\s+i|to)\s+invest\s+my\s+money\b",
    r"\b(?:best|top)\s+(?:mutual\s+)?(?:fund|funds|scheme|schemes)\b",
    r"\b(?:guaranteed|assured)\s+return[s]?\b",
    r"\b(?:is\s+it\s+good\s+time\s+to\s+(?:invest|buy|sell))\b",
    r"\bhelp\s+me\s+choose\s+between\b",
]
ADVICE_REGEX = re.compile("|".join(ADVICE_PATTERNS), re.IGNORECASE)

ABUSE_PATTERNS = [
    r"\bfuck\b", r"\bbitch\b", r"\bshit\b", r"\basshole\b", r"\bcrap\b", r"\bdamn\b", r"\bidiot\b", r"\bstupid\b"
]
ABUSE_REGEX = re.compile("|".join(ABUSE_PATTERNS), re.IGNORECASE)


class GrowwRAGEngine:
    def __init__(self, sources_path: str = "sources.csv", corpus_path: str = "sample_qa.txt"):
        self.sources_path = sources_path
        self.corpus_path = corpus_path
        self.sources: Dict[str, Dict[str, str]] = {}
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self.load_data()

    def load_data(self) -> None:
        if os.path.exists(self.sources_path):
            with open(self.sources_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_id = row.get("source_id", row.get("id", ""))
                    if source_id:
                        self.sources[source_id] = row

        if os.path.exists(self.corpus_path):
            try:
                if self.corpus_path.endswith(".json"):
                    with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                        self.documents = json.load(f)
                else:
                    with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                        content = f.read()
                    blocks = re.split(r'\n\s*\n', content)
                    parsed_docs = []
                    for idx, block in enumerate(blocks):
                        if not block.strip():
                            continue
                        lines = [l.strip() for l in block.split('\n') if l.strip()]
                        title = lines[0] if lines else f"FAQ Chunk {idx+1}"
                        body = "\n".join(lines[1:]) if len(lines) > 1 else title
                        parsed_docs.append({
                            "source_id": f"faq_{idx+1}",
                            "title": title,
                            "category": "Mutual Funds FAQ",
                            "organization": "Groww",
                            "content": body,
                            "url": GROWW_EDU_URL
                        })
                    if parsed_docs:
                        self.documents = parsed_docs
            except Exception:
                pass

        if not self.documents:
            self.documents = DEFAULT_MUTUAL_FUND_CORPUS

        if self.documents:
            corpus_texts = [
                f"{doc.get('title', '')} {doc.get('category', '')} {doc.get('content', '')}"
                for doc in self.documents
            ]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
            self.doc_vectors = self.vectorizer.fit_transform(corpus_texts)

    def detect_pii(self, query: str) -> Optional[str]:
        detected = []
        if PAN_REGEX.search(query):
            detected.append("PAN")
        if AADHAAR_REGEX.search(query):
            detected.append("Aadhaar Number")
        if PHONE_REGEX.search(query):
            detected.append("Phone Number")
        if EMAIL_REGEX.search(query):
            detected.append("Email Address")
        return ", ".join(detected) if detected else None

    def is_advice_or_opinion_seeking(self, query: str) -> bool:
        return bool(ADVICE_REGEX.search(query.strip()))

    def is_abusive(self, query: str) -> bool:
        return bool(ABUSE_REGEX.search(query.strip()))

    def _generate_suggested_questions(self, query: str, retrieved: List[Dict[str, Any]]) -> List[str]:
        q_lower = query.lower()
        if "sip" in q_lower:
            return [
                "What is rupee-cost averaging in SIP investments?",
                "Can I automate my monthly mutual fund SIP deductions?",
                "What is the minimum amount required to start an SIP?"
            ]
        if "elss" in q_lower or "tax" in q_lower:
            return [
                "How do I claim Section 80C deductions for ELSS investments?",
                "What happens to my ELSS mutual fund units after the 3-year lock-in?",
                "Can I start a monthly SIP in an ELSS tax saver fund?"
            ]
        if "cut-off" in q_lower or "timing" in q_lower or "nav" in q_lower:
            return [
                "What is the 3:00 PM cut-off rule for equity fund same-day NAV?",
                "What are the purchase cut-off timings for liquid and overnight funds?",
                "How does bank fund realization impact the NAV allotment date?"
            ]
        return [
            "What is a Systematic Investment Plan (SIP)?",
            "What is the statutory lock-in period for ELSS mutual funds?",
            "What are the cut-off timings to get same-day NAV on purchases?"
        ]

    def retrieve_multi_chunks(self, clean_query: str, top_k: int = 3, threshold: float = 0.02) -> List[Dict[str, Any]]:
        """Retrieves top matching chunks based on actual cosine similarity above a minimum threshold."""
        if not self.documents or self.doc_vectors is None or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        scored_docs = []
        for idx, doc in enumerate(self.documents):
            score = float(similarities[idx])
            if score >= threshold:
                scored_docs.append({
                    "doc": doc,
                    "score": score,
                    "index": idx
                })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def answer_query(self, query: str, filter_org: Optional[str] = None) -> Dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please enter a specific mutual fund inquiry or financial topic.",
                "overview": "Please enter a specific mutual fund inquiry or financial topic.",
                "sources": [],
                "status": "empty",
                "score": 0.0
            }

        if self.is_abusive(clean_query):
            msg = "Please maintain respectful and appropriate language while interacting with the Groww Mutual Fund FAQ Assistant."
            return {"answer": msg, "sources": [], "status": "refused_abuse", "score": 0.0}

        if self.detect_pii(clean_query):
            msg = "For your privacy and security, do not share personal information. This system does not store sensitive documents."
            return {"answer": msg, "sources": [], "status": "refused_pii", "score": 0.0}

        if self.is_advice_or_opinion_seeking(clean_query):
            msg = "I cannot provide personalized investment advice or recommendations. Please consult a SEBI-registered financial advisor."
            return {"answer": msg, "sources": [], "status": "refused_advice", "score": 0.0}

        retrieved = self.retrieve_multi_chunks(clean_query, top_k=3, threshold=0.02)

        if not retrieved:
            fallback = "I don't have this information in my official sources. Please ask questions related to Groww mutual funds, SIPs, ELSS, cut-off timings, or direct plans."
            return {"answer": fallback, "sources": [], "status": "out_of_scope", "score": 0.0}

        overview_sections = [item["doc"].get("content", "") for item in retrieved]
        detailed_overview = "\n\n".join(overview_sections)
        
        sources_list = [{
            "title": item["doc"].get("title"),
            "url": item["doc"].get("url", GROWW_EDU_URL),
            "organization": item["doc"].get("organization", "Groww"),
            "category": item["doc"].get("category", "General")
        } for item in retrieved]

        suggestions = self._generate_suggested_questions(clean_query, retrieved)

        formatted_answer = f"### Detailed Overview\n{detailed_overview}\n\n### Exhaustive Sources\n"
        for s in sources_list:
            formatted_answer += f"- [{s['title']}]({s['url']}) · *{s['organization']} ({s['category']})*\n"
        formatted_answer += f"\n### Suggested Questions\n"
        for sug in suggestions:
            formatted_answer += f"- {sug}\n"

        return {
            "answer": formatted_answer,
            "overview": detailed_overview,
            "sources": sources_list,
            "suggestions": suggestions,
            "status": "ok",
            "score": retrieved[0]["score"]
        }