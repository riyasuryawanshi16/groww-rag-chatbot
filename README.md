# Groww Mutual Funds RAG-Based FAQ Chatbot 📈

A production-ready, guardrailed Retrieval-Augmented Generation (RAG) FAQ Chatbot for **Groww Mutual Funds**. Grounded in official documentation from **Groww**, **SEBI**, **AMFI**, and **SBI Mutual Fund** (spanning Large-Cap, Flexi-Cap, and ELSS schemes).

---

## 🛡️ Core Guardrails & System Constraints

This chatbot enforces strict regulatory and safety constraints:

1. **Facts-Only Retrieval**: Answers are derived exclusively from verified official documentation. If a query falls outside the official corpus, the system explicitly refuses rather than hallucinating.
2. **Length Constraint (≤ 3 Sentences)**: Every response is programmatically bounded to **at most three sentences**, delivering concise, high-density factual answers.
3. **Mandatory Source Citations**: Every single response includes a verified, direct markdown hyperlink to the original official source document.
4. **Investment Advice Safe Refusal**: If a user asks for stock/fund recommendations, portfolio reviews, or "which fund should I buy/sell", the chatbot safely refuses and redirects the user with an official educational link to [Groww Mutual Funds Hub](https://groww.in/mutual-funds).
5. **PII Shield**: Regular expressions intercept and block sensitive Personal Identifiable Information (PAN, Aadhaar numbers, phone numbers, and email addresses) before any query is processed or recorded.
6. **Prominent UI Disclaimer**: Displays **"Facts-only. No investment advice."** prominently at the top of the user interface and on the sidebar.

---

## 🏗️ Architecture & Data Flow

```text
               ┌───────────────────────────────┐
               │    User Query (Streamlit)     │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │        PII & Identity Shield        │
            │   (PAN, Aadhaar, Phone, Email)      │
            └───────┬─────────────────────┬───────┘
                    │ [PII Detected]      │ [Clean Query]
                    ▼                     ▼
          ┌───────────────────┐ ┌─────────────────────────────┐
          │  Security Rejection│ │ Investment Advice Detector  │
          │  (Data Protected) │ │ ("should I buy", picks, etc)│
          └───────────────────┘ └──────┬───────────────┬──────┘
                                       │ [Advice]      │ [FAQ Query]
                                       ▼               ▼
                             ┌────────────────┐ ┌───────────────────────────┐
                             │ Safe Refusal + │ │ TF-IDF Vector Space Index │
                             │ Groww Edu Link │ │ Cosine Similarity Search  │
                             └────────────────┘ └─────────────┬─────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ Top-Passage Retrieval     │
                                                │ Threshold Score Filtering │
                                                └─────────────┬─────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ Extractive Sentence Score │
                                                │ Strict ≤ 3 Sentences      │
                                                └─────────────┬─────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ Mandatory Citation Append │
                                                │ Formatted Markdown Output │
                                                └───────────────────────────┘
```

---

## 📚 Official Knowledge Corpus (21 URLs)

The dataset in `sources.csv` and `data/corpus.json` comprises 21 high-authority public URLs:

| Source ID | Organization | Category | Title | Official URL |
|---|---|---|---|---|
| `SRC01` | **Groww** | Basics | Groww Mutual Funds Hub | [groww.in/mutual-funds](https://groww.in/mutual-funds) |
| `SRC02` | **Groww** | Basics | What is a Mutual Fund | [groww.in/p/mutual-funds/what-is-mutual-fund](https://groww.in/p/mutual-funds/what-is-mutual-fund) |
| `SRC03` | **Groww** | Basics | Systematic Investment Plan (SIP) | [groww.in/p/mutual-funds/sip-systematic-investment-plan](https://groww.in/p/mutual-funds/sip-systematic-investment-plan) |
| `SRC04` | **Groww** | Basics | Net Asset Value (NAV) Explained | [groww.in/p/mutual-funds/nav-net-asset-value](https://groww.in/p/mutual-funds/nav-net-asset-value) |
| `SRC05` | **Groww** | Basics | Expense Ratio in Mutual Funds | [groww.in/p/mutual-funds/expense-ratio](https://groww.in/p/mutual-funds/expense-ratio) |
| `SRC06` | **Groww** | Large-Cap | Large Cap Mutual Funds Guide | [groww.in/p/mutual-funds/large-cap-funds](https://groww.in/p/mutual-funds/large-cap-funds) |
| `SRC07` | **Groww** | Flexi-Cap | Flexi Cap Mutual Funds Guide | [groww.in/p/mutual-funds/flexi-cap-funds](https://groww.in/p/mutual-funds/flexi-cap-funds) |
| `SRC08` | **Groww** | ELSS | ELSS Tax Saving Mutual Funds | [groww.in/p/mutual-funds/elss-tax-saving-mutual-funds](https://groww.in/p/mutual-funds/elss-tax-saving-mutual-funds) |
| `SRC09` | **Groww** | Operations | Cut-off Timings for Mutual Funds | [groww.in/p/mutual-funds/cut-off-timings-for-mutual-funds](https://groww.in/p/mutual-funds/cut-off-timings-for-mutual-funds) |
| `SRC10` | **Groww** | Operations | Exit Load in Mutual Funds | [groww.in/p/mutual-funds/exit-load-in-mutual-funds](https://groww.in/p/mutual-funds/exit-load-in-mutual-funds) |
| `SRC11` | **SEBI** | Regulations | SEBI FAQs on Mutual Funds | [sebi.gov.in/mutual_funds_faq.html](https://www.sebi.gov.in/sebi_data/faqfiles/mutual_funds_faq.html) |
| `SRC12` | **SEBI** | Regulations | Categorization of Mutual Fund Schemes | [sebi.gov.in/.../categorization-and-rationalization](https://www.sebi.gov.in/legal/circulars/oct-2017/categorization-and-rationalization-of-mutual-fund-schemes_36199.html) |
| `SRC13` | **SEBI** | Regulations | SEBI Investor Education Portal | [investor.sebi.gov.in/mutual-fund.html](https://investor.sebi.gov.in/investor-education/mutual-fund.html) |
| `SRC14` | **SEBI** | Regulations | SEBI Risk-o-meter & Investor Guidelines | [sebi.gov.in/.../investor-protection-guidelines](https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/investor-protection-guidelines_67382.html) |
| `SRC15` | **AMFI** | Basics | AMFI What are Mutual Funds | [amfiindia.com/.../what-are-mutual-funds](https://www.amfiindia.com/investor-corner/knowledge-center/what-are-mutual-funds.html) |
| `SRC16` | **AMFI** | Classification | AMFI Types of Mutual Fund Schemes | [amfiindia.com/.../types-of-funds](https://www.amfiindia.com/investor-corner/knowledge-center/types-of-funds.html) |
| `SRC17` | **AMFI** | Operations | AMFI KYC Guidelines for Investors | [amfiindia.com/.../kyc](https://www.amfiindia.com/investor-corner/knowledge-center/kyc.html) |
| `SRC18` | **AMFI** | Regulations | AMFI Risk-o-meter Guidelines | [amfiindia.com/.../risk-o-meter](https://www.amfiindia.com/investor-corner/knowledge-center/risk-o-meter.html) |
| `SRC19` | **SBI MF** | Large-Cap | SBI Bluechip Fund Scheme Details | [sbimf.com/.../sbi-bluechip-fund](https://www.sbimf.com/en-us/equity-schemes/sbi-bluechip-fund) |
| `SRC20` | **SBI MF** | Flexi-Cap | SBI Flexicap Fund Scheme Details | [sbimf.com/.../sbi-flexicap-fund](https://www.sbimf.com/en-us/equity-schemes/sbi-flexicap-fund) |
| `SRC21` | **SBI MF** | ELSS | SBI Long Term Equity Fund (ELSS) | [sbimf.com/.../sbi-long-term-equity-fund](https://www.sbimf.com/en-us/equity-schemes/sbi-long-term-equity-fund) |

---

## 📂 Project Directory Structure

```text
groww-mf-faq-bot/
├── app.py              # Streamlit Web Application with Groww emerald UI
├── rag_engine.py       # Core RAG retrieval, PII filter, advice refusal, sentence limiter
├── test_rag.py         # Automated unit test suite verifying all guardrails
├── sources.csv         # 21 verified official public URLs with metadata
├── sample_qa.txt       # Evaluation file with 9 comprehensive query/answer pairs
├── requirements.txt    # Pinned dependencies
├── data/
│   └── corpus.json     # Verified factual knowledge passages mapped 1:1 to sources.csv
└── README.md           # Project documentation and architectural guide
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Clone or Navigate to the Project
```bash
cd C:\Users\Asus\.gemini\antigravity-ide\scratch\groww-mf-faq-bot
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Automated Tests

Run the comprehensive unit test suite:
```bash
python test_rag.py
```
This validates:
- [x] Correct URL count (between 15 and 25) and column schema in `sources.csv`.
- [x] Indian PAN, Aadhaar, and phone number PII shield.
- [x] Investment advice refusal and educational redirect link.
- [x] Strict sentence count limit (≤ 3 sentences).
- [x] Out-of-scope non-financial question deflections.
- [x] Factual precision on ELSS lock-in and cut-off timings.

---

## 💻 Running the Streamlit Web Application

Launch the Streamlit interface locally:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Interactive Features:
- **Banner Disclaimer**: Prominent regulatory banner stating *"Facts-only. No investment advice."*
- **3 Suggestion Chips**: One-click FAQ queries:
  1. *What is the lock-in period and tax benefit of an ELSS fund?*
  2. *What are the cut-off timings for equity mutual fund purchases?*
  3. *What is the difference between Large-Cap and Flexi-Cap funds under SEBI rules?*
- **Safety Test Buttons**: Quickly test how the chatbot intercepts investment advice queries and shields personal identification numbers.
- **Authority Filter**: Filter responses by Groww, SEBI, AMFI, or SBI Mutual Fund.
- **Source Catalog Explorer**: Inspect all 21 URLs and descriptions inside the sidebar drawer.

---

## ⚠️ System Scope & Limitations

1. **Facts-Only Scope**: The system provides factual definitions, operational mechanics, and regulatory norms. It **never** produces buy/sell recommendations, return forecasts, or individual portfolio ratings.
2. **Deterministic Fallback**: In the absence of high-confidence matches in the official corpus, the chatbot rejects the prompt rather than synthesizing assumptions.
3. **Offline Knowledge Baseline**: In this reference implementation, passages are pre-indexed in `data/corpus.json` to guarantee instant zero-setup responsiveness without requiring paid cloud API keys or external scraper dependencies.
4. **Production Scaling**:
   - For enterprise scale, `rag_engine.py` can be extended with dense embedding models (e.g. `sentence-transformers/all-MiniLM-L6-v2`) and persistent vector stores like ChromaDB or FAISS.
   - Live scraping pipelines can periodically refresh `sources.csv` to capture updated NAV cut-off regulations or tax revisions.
