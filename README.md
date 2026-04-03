# PhD Application Agent

A multi-agent AI pipeline that scrapes PhD positions, scores how well your CV matches them, and generates tailored cover letter sections — powered by Claude and LangGraph.

---

## How it works

```
CLI Input (--url, --cv)
        │
        ▼
  ┌─────────────┐
  │  Supervisor │  pre-loads CV text into SharedState
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │ Scraper Agent│  fetch → score → save to Excel
  └──────┬───────┘
         │
    match_score ≥ 50?
    ┌────┴──────────────┐
   YES                  NO
    │                   │
    ▼                   ▼
┌──────────────┐   end_node
│ Writer Agent │◄──────────── retrieve similar letters
│              │──────────── store generated letter
└──────┬───────┘      │
       │        ┌─────┴──────┐
       │        │  ChromaDB  │
       │        │ (RAG store)│
       │        └────────────┘
       ▼
   end_node
   (cover letter saved)
```

**Supervisor** pre-extracts the CV from PDF directly into `SharedState` before anything else runs. This avoids a subtle but critical bug: embedding raw CV text inside a JSON string (as the scraper's final response) breaks `json.loads` on newlines and special characters like ligatures. The CV travels PDF → SharedState → Writer, never through JSON.

**Scraper Agent** runs its own internal ReAct loop with 4 tools. When done, it returns a compact JSON containing only safe string fields (title, university, description, requirements, score). It writes to `phd_positions.xlsx` with colour-coded match scores.

**Writer Agent** runs its own internal ReAct loop with 5 tools. It queries ChromaDB for semantically similar past cover letters, uses them as style context, generates 2–3 tailored paragraphs, humanizes via Rephrasy API, saves to a `.txt` file, and stores the result back in ChromaDB for future retrieval.

---

## Features

- **CV-position match scoring** — Claude scores 4 weighted categories (technical skills 30%, research experience 30%, domain alignment 25%, education 15%) and returns a 0–100 score with verdict, strengths, and gaps
- **Conditional routing** — Writer Agent only runs if score ≥ 50; below that threshold only the Excel database is updated
- **RAG cover letter generation** — ChromaDB with local `all-MiniLM-L6-v2` embeddings retrieves your most similar past letters as style references; improves with every run
- **Humanization** — Rephrasy API post-processes generated text; falls back to original if API fails
- **Excel database** — `phd_positions.xlsx` with colour-coded scores (green ≥75, yellow ≥50, red <50), duplicate detection, and full position metadata
- **Retry on rate limits** — `tenacity` exponential backoff (5s → 60s, 5 attempts) on all Claude API calls
- **Smart filenames** — output `.txt` derived from URL slug automatically; override with `--output`

---

## Project structure

```
phd-agent/
├── apply_agent.py          # entire pipeline — one file
├── templates/
│   └── cv.pdf            # your CV (default path)
├── chroma_db/            # local vector store (auto-created)
├── phd_positions.xlsx    # position database (auto-created)
└── .env                  # API keys
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/yahyamomtaz/apply_agent.git
cd phd-agent
pip install langchain-anthropic langgraph playwright beautifulsoup4 \
            pypdf2 openpyxl requests chromadb sentence-transformers tenacity python-dotenv
playwright install chromium
```

**2. Create `.env`**

```env
ANTHROPIC_API_KEY=sk-ant-...
REPHRASY_API_KEY=...          # optional — humanizer falls back gracefully if missing
```

**3. Place your CV**

```bash
mkdir templates
cp /path/to/your/cv.pdf templates/cv.pdf
```

---

## Usage

```bash
# Full run — scrape, score, generate cover letter
python apply_agent.py --url "https://academicpositions.com/ad/phd-position-123"

# Custom CV path
python apply_agent.py --url "https://..." --cv path/to/my_cv.pdf

# Score only — no cover letter generated
python phd_agent.py --url "https://..." --no-cover-letter

# Custom output filename
python phd_agent.py --url "https://..." --output eth_zurich_cover_letter.txt
```

### Example output

```
════════════════════════════════════════════════════════
PhD APPLICATION ASSISTANT  —  Multi-Agent
════════════════════════════════════════════════════════
URL       : https://academicpositions.com/ad/...
CV        : templates/cv.pdf
Mode      : Scrape + Score + Generate
RAG store : 3 past letter(s)
Output    : phd_position_nlp_cover_letter.txt
════════════════════════════════════════════════════════

SUPERVISOR — dispatching to Scraper Agent
Supervisor pre-loaded CV (4821 chars)

SCRAPER AGENT
Fetching: https://...
Scoring match for: PhD in NLP...
Match score: 78/100 — Excellent Match

SUPERVISOR — score 78/100 ≥ 50, dispatching to Writer Agent

WRITER AGENT
Retrieving similar letters...   (3 found, top similarity 88.4%)
Generating cover letter (with RAG)...
Humanizing...
Cover letter saved to: phd_position_nlp_cover_letter.txt
Stored in RAG (total: 4 letters)

════════════════════════════════════════════════════════
📋 FINAL SUMMARY
════════════════════════════════════════════════════════
Position  : PhD in Natural Language Processing
University: ETH Zurich
Score     : 78/100 — Excellent Match
Cover letter: ✅ phd_position_nlp_cover_letter.txt
RAG stored  : ✅
════════════════════════════════════════════════════════
```

---

## Architecture details

### SharedState

All data flows through a single `SharedState` TypedDict — no message passing between agents.

| Field | Set by | Read by |
|---|---|---|
| `url`, `cv_path`, `output_filename`, `no_cover_letter` | CLI | Supervisor |
| `cv_text` | Supervisor (PDF extraction) | Scraper, Writer |
| `position_title`, `position_university` | Scraper | Writer, end_node |
| `position_description`, `position_requirements` | Scraper | Writer |
| `match_score`, `match_verdict`, `match_recommendation` | Scraper | Supervisor routing, end_node |
| `cover_letter_filename`, `rag_stored` | Writer | end_node |

### Why CV text is pre-loaded in the Supervisor

The scraper's final step is to return a JSON string that LangGraph parses to populate SharedState. If `cv_text` were included in that JSON, the raw PDF text — which contains newlines, Unicode ligatures (`ﬁ`, `ﬀ`), and curly quotes — would silently break `json.loads`, causing the match score to fall back to `-1` and the writer to be skipped. Pre-loading in the Supervisor means the CV travels Python object → SharedState and never touches JSON serialization.

### Match scoring weights

| Category | Weight | What it measures |
|---|---|---|
| Technical skills | 30% | Programming languages, tools, methods from requirements |
| Research experience | 30% | Publications, projects, lab experience |
| Domain alignment | 25% | Field match between CV and position focus |
| Education | 15% | Degree level, institution, relevant coursework |

### RAG embeddings

Local `all-MiniLM-L6-v2` via `sentence-transformers` — no external API, no cost. First run downloads ~90 MB model to local cache. ChromaDB stores embeddings in `./chroma_db/` with cosine similarity. Retrieves top-3 most similar past letters as style context for generation; improves every time a new letter is generated and stored.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-anthropic` | Claude LLM integration |
| `langgraph` | Multi-agent graph orchestration |
| `playwright` | Headless browser for scraping |
| `beautifulsoup4` | HTML parsing and cleaning |
| `pypdf2` | CV PDF extraction |
| `openpyxl` | Excel database |
| `chromadb` | Local vector store |
| `sentence-transformers` | Local embeddings (all-MiniLM-L6-v2) |
| `tenacity` | Retry with exponential backoff |
| `requests` | Rephrasy humanizer API |
| `python-dotenv` | Environment variable loading |

---

## Limitations

- Rephrasy API free tier has character limits — falls back gracefully to un-humanized text
- Claude Sonnet rate limits on free/low-tier API keys may slow down runs with large CVs or descriptions
- `match_score_position` calls Claude directly (not via tool loop) so it counts against the same rate limit
