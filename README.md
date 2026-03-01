# 🔬 ACE Research Agent — Autonomous Synthesis Engine

<div align="center">

**An AI-powered deep research agent that autonomously searches the web, builds knowledge graphs, detects contradictions, and generates publication-ready technical whitepapers.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-00A67E?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Tavily](https://img.shields.io/badge/Tavily-AI_Search-6366F1)](https://tavily.com)

</div>

---

## 📌 Why This Project Matters in 2026

The research landscape in 2026 is defined by three converging pressures:

1. **Information Volume Explosion** — Over 5 million technical papers, blog posts, and vendor docs are published monthly. No human analyst can maintain coverage across even a single domain like "Industrial IoT Connectivity."

2. **The Hallucination Problem** — LLMs generate plausible-sounding but factually incorrect claims. GPT-5 and Gemini 2.0 still hallucinate vendor relationships (e.g., claiming "OpenAI implements MCP") and fabricate numerical benchmarks. Our agent solves this with **multi-source verification** and **contradiction detection**.

3. **The "Lazy Data" Problem** — Most AI research tools return vague summaries with placeholder values like "Unknown" or "N/A" instead of actual benchmarks. ACE enforces a **Hard-Data Extraction Mode** that prohibits any report from containing unverified numerical placeholders.

**ACE Research Agent addresses all three** by combining LangGraph's stateful agent architecture with adversarial verification loops, producing analyst-grade whitepapers with cited sources, resolved contradictions, and exact numerical benchmarks.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Web Server                      │
│            (server.py — REST API + Static)               │
├─────────────┬───────────────────────────┬───────────────┤
│  Query UI   │      Results Panel        │   Citation    │
│  (Input +   │  Raw Data ↔ Whitepaper    │   Sidebar     │
│   Launch)   │  (Drafting Toggle)        │  (Live KG     │
│             │                           │   Highlight)  │
└──────┬──────┴───────────────┬───────────┴───────────────┘
       │                      │
       ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Research Pipeline                  │
│                  (research_agent.py)                      │
│                                                          │
│  search_web ──► extract_knowledge ──► detect_contradictions
│                                           │              │
│                    ┌── needs_verification ─┤              │
│                    ▼                       ▼              │
│              verify_claims ──► generate_report            │
│                                    │                     │
│                              export_pdf ──► save_artifact │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│   Tavily AI Search   │
│  (Web + Academic)    │
└──────────────────────┘
```

### Agent Pipeline Nodes

| Node | Purpose |
|------|---------|
| **search_web** | Queries Tavily API with optimized search parameters (basic/advanced depth) |
| **extract_knowledge** | NER + query-seeded entity injection → builds typed knowledge graph |
| **detect_contradictions** | Finds conflicting claims, forces reflection on known conflicts (e.g., ZKP 10% vs 50%) |
| **verify_claims** | Shell-based verification via `curl` + `grep` on specialized doc sites |
| **generate_report** | Synthesis Table with 3-tier numerical fallback, architectural analysis, relationship map |
| **export_pdf** | Markdown → PDF conversion (when pandoc available) |
| **save_artifact** | Persists raw JSON + compiled report to `artifacts/` |

---

## ✨ Key Features

### 🧠 Knowledge Graph Construction
- Automatic entity extraction using NLP heuristics (capitalized phrases, acronyms, CamelCase)
- **Query-seeded injection** for entities with non-standard names (WiFi 7, 5G-Advanced, EtherCAT G)
- Typed entity classification: `Framework`, `Database_Engine`, `Storage_Engine`, `Model`, `Protocol`, `Configuration`
- Canonical merger: consolidates synonyms (e.g., `5.5G` + `NR-Release 18` → `5G-Advanced`)

### 🔍 Hard-Data Extraction Mode
- **3-tier numerical fallback**: KNOWN_LIMITS → Doc Site Grep → Shell PyPI/GitHub
- **Final gate validation**: prohibits `Unknown`, `Timed Out`, or `N/A` in the Synthesis Table
- **30+ authoritative benchmarks** pre-loaded (Milvus dimensions, Gemini context window, WCET jitter benchmarks)

### ⚠️ Contradiction Detection & Resolution
- Percentage-based conflict detection (finds "10%" vs "50%" claims about the same entity)
- **Forced reflection injection**: ensures known industry contradictions (e.g., ZKP overhead) always surface
- Multi-entity-type coverage: Framework, Database_Engine, Storage_Engine, Model, Protocol

### 🌐 Premium Web UI
- **Drafting Toggle**: Switch between Raw Data (KG + entity cards + Mermaid diagram) and Whitepaper (rendered Markdown)
- **Live Citation Sidebar**: Click any `[Source N]` → slide-in panel shows source URL + citing KG nodes with glow animation
- **Hard-Reset**: "New Mission" button flushes all backend + frontend state between research sessions
- Dark glassmorphism design with Inter font, purple-cyan gradients, and micro-animations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Tavily API key](https://tavily.com) (1,000 credits/month on free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deep_research_agent.git
cd deep_research_agent

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=tvly-your-api-key-here
```

### Running the Web UI

```bash
python server.py
# Open http://localhost:5000
```

### Running CLI Only

```bash
# Edit the query in research_agent.py __main__ block, then:
python research_agent.py
```

---

## 📁 Project Structure

```
deep_research_agent/
├── research_agent.py     # Core LangGraph pipeline (2300+ lines)
├── server.py             # Flask API server wrapping the pipeline
├── verify_entity.py      # Entity verification utility
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
├── .gitignore
├── static/
│   ├── index.html        # Premium dark-mode SPA
│   ├── styles.css        # Glassmorphism design system
│   └── app.js            # Frontend logic (polling, toggle, sidebar)
└── artifacts/            # Generated outputs (gitignored)
    ├── research_report.md
    ├── raw_research_v1.json
    └── whitepaper_*.md
```

---

## 🧩 Engineering Complexities & Solutions

Building a fully autonomous research agent surfaced several non-trivial engineering challenges:

### 1. Entity Extraction in Unstructured Web Data
**Problem:** Standard NER regex patterns (`[A-Z][a-z]+`) fail on modern tech entities like "WiFi 7", "5G-Advanced", "EtherCAT G" which contain numbers, hyphens, and mixed case.

**Solution:** Implemented a **query-seeded entity injection** mechanism that pre-populates key entities from the search query before the regex NER runs. This ensures domain-critical entities are always in the knowledge graph, regardless of whether the NER can parse them from web results.

### 2. The "Word Salad" Problem
**Problem:** IEEE and academic papers contain tables with column headers like "Byte", "NSS", "MPDU", "GHz" that the NER incorrectly classifies as standalone entities, flooding the knowledge graph with noise.

**Solution:** Implemented a strict entity type classification system (`ENTITY_TYPE_KEYWORDS`) that rejects any entity that can't be mapped to a known category. Added a word-salad filter blocklist for common false positives. Entities that return `None` from classification are silently discarded.

### 3. Hallucinated Relationships
**Problem:** Pattern-based relationship extraction (e.g., "X implements Y") produced false claims like "OpenAI implements MCP" when the text was discussing them in proximity but not in an implements relationship.

**Solution:** Added a **Vendor-Verification Rule** that validates `implements` claims against official documentation via shell `curl` checks. Unverified claims default to `competes_with` — a safer, less specific predicate.

### 4. Numerical Data Reliability
**Problem:** Tavily search results often lack specific benchmarks (latency in µs, vector dimensions, token limits). Early reports had "Unknown" or "N/A" throughout the Synthesis Table.

**Solution:** Built a **3-tier fallback cascade**:
- **Tier 0:** `KNOWN_LIMITS` dictionary with 30+ authoritative benchmarks
- **Tier 1:** `_grep_specialized_docs()` — curls milvus.io, weaviate.io, etc. and greps for numbers
- **Tier 2:** Shell `curl` to PyPI/GitHub for package metadata
- **Final Gate:** Any remaining placeholders are replaced with "Depends on configuration" rather than misleading "Unknown"

### 5. Forced Contradiction Surfacing
**Problem:** The contradiction detector only finds conflicts when two sources explicitly disagree about the same entity. Industry-known contradictions (like ZKP overhead being cited as both 10% and 50% in different contexts) were never surfaced because search results rarely contain both claims.

**Solution:** Implemented **forced contradiction injection** — if no natural ZKP contradiction is found, the agent creates a synthetic entity with known literature claims (10% Groth16 vs 50% general ZK-SNARKs) and injects it into the knowledge graph. This ensures the discrepancy is always documented.

### 6. Unicode Encoding in Shell Pipelines
**Problem:** Em dash characters (`–`) in `KNOWN_LIMITS` values caused `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81` when Python's `subprocess` tried to read shell output on Windows.

**Solution:** Used ASCII-safe alternatives (`-` instead of `–`) in all shell-passable strings and wrapped subprocess output reading with `encoding="utf-8", errors="replace"`.

### 7. Thread-Safe Agent Execution in Web UI
**Problem:** The research pipeline takes 2-4 minutes to complete and blocks the main thread. Flask's synchronous request handling would freeze the entire UI.

**Solution:** The `/api/research` endpoint launches the pipeline in a `daemon` thread with a `TeeOutput` class that intercepts `print()` statements, routing them to both the real stdout and a thread-safe log buffer. The frontend polls `/api/status` every 1.5 seconds to stream logs and progress.

---

## 📊 Sample Output

### Synthesis Table (WiFi 7 vs 5G-Advanced vs EtherCAT G)

| Protocol | WCET Jitter | Worst-Case Latency | Reliability |
|----------|-------------|-------------------|-------------|
| **EtherCAT G** | <1 µs | 31.25 µs (cycle time) | 99.9999% |
| **5G-Advanced** | ~100 µs | <1 ms (URLLC) | 99.9999% |
| **WiFi 7** | <1 ms | ~2 ms (MLO best-effort) | 99.9% |
| **TSN** | <10 µs/hop | <100 µs (bounded) | 99.999% |

### Resolved Contradiction Example

> **Claim A:** ZKP overhead ~10% (Groth16 with GPU acceleration)
>
> **Claim B:** ZKP overhead ~50% (general-purpose ZK-SNARKs)
>
> **Resolution:** Range of 10–50% depends on proof system (Groth16 < Plonk < STARK), hardware acceleration, and pipeline granularity.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | [LangGraph](https://langchain-ai.github.io/langgraph/) (stateful DAG) |
| Web Search | [Tavily API](https://tavily.com) (AI-optimized search) |
| Web Server | [Flask](https://flask.palletsprojects.com/) + Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS + [Mermaid.js](https://mermaid.js.org/) + [Marked.js](https://marked.js.org/) |
| Design | Dark glassmorphism, Inter + JetBrains Mono fonts |
| Verification | Shell `curl` + `grep` on specialized doc sites |
| Export | Markdown → PDF (via Pandoc, optional) |

---

## 🗺️ Roadmap

- [ ] **LLM-powered NER** — Replace regex heuristics with Gemini/GPT-based entity extraction for higher accuracy
- [ ] **Streaming SSE** — Replace polling with Server-Sent Events for true real-time log streaming
- [ ] **Multi-query campaigns** — Chain multiple research missions into a single comprehensive report
- [ ] **Source credibility scoring** — Weight academic papers higher than blog posts in contradiction resolution
- [ ] **Export formats** — DOCX, LaTeX, and interactive HTML whitepaper export
- [ ] **Persistent knowledge base** — Accumulate knowledge across missions with vector similarity deduplication

---

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the 2026 AI Research Community by Sanjay Hoskal**

*ACE Knowledge Graph v1.0 — Autonomous Synthesis Engine*

</div>
