# Sephora Skincare Reviews — Stats + RAG Chatbot

A Streamlit app that answers two kinds of questions about Sephora skincare
product reviews:

- **Stat-style questions** ("What are the most common complaints?", "What's
  most common in June for Moisturizers?") are answered from precomputed issue
  counts, tagged either with fast keyword matching or via the Groq API.
- **Open-ended questions** ("Why do people say this didn't work for them?")
  are answered with retrieval-augmented generation (RAG) over a semantic
  search index of a cached review sample.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: sidebar setup, stats dashboard, and chat. |
| `sephora_core.py` | Shared logic: sampling, issue tagging, aggregates, the vector index, and question routing/answering. Framework-agnostic so it's easy to test or reuse from a script/CLI. |
| `sample_1k.parquet` | A cached 1,000-review sample (already tagged-ready), committed to the repo so the app runs out of the box without the full ~1M-row raw dataset. |
| `tests/test_core.py` | Unit tests for the pure logic in `sephora_core.py` (no network/API calls). |

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at <https://console.groq.com/keys>, then set it as an
environment variable before running locally:

```bash
# Mac/Linux
export GROQ_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-key-here"
```

If deploying on Streamlit Community Cloud, add `GROQ_API_KEY` under **Secrets**
in the app's settings instead — never commit the key to GitHub. The app reads
`st.secrets["GROQ_API_KEY"]` automatically if the environment variable isn't
set.

## Run

```bash
streamlit run app.py
```

Then open the browser tab it pops up (usually <http://localhost:8501>).

On a cloud host, or if you don't have the raw dataset, just click
**Build / Rebuild pipeline** in the sidebar — it will use the committed
`sample_1k.parquet` automatically.

### Using the full raw dataset (optional, local only)

If you have the raw Sephora dataset (`product_info.csv` + one or more
`reviews*.csv` files, e.g. from the [Kaggle Sephora reviews
dataset](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)),
point the sidebar's "Data folder" field at that directory and click
**Build / Rebuild pipeline**. This resamples `sample_1k.parquet` from the raw
data, tags issues, and rebuilds the vector index. The raw dataset itself is
intentionally not committed to the repo (too large for GitHub).

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The tests cover the pure logic in `sephora_core.py` (keyword tagging, month
and category extraction, stat-vs-RAG routing, sampling, and aggregation) and
run without a `GROQ_API_KEY` or network access.

## Notes on issue tagging

- **Keyword tagging** (default) is instant and free but relies on a curated
  keyword/stem list (`ISSUE_KEYWORDS` in `sephora_core.py`). Words are matched
  as whole words or word-stems, so e.g. `"cap"` doesn't fire on `"capacity"`.
- **LLM-based tagging** (toggle in the sidebar) sends batches of negative
  reviews to the Groq API for more accurate, phrasing-agnostic tagging. It's
  slower and uses more API calls, but results are cached under `.cache/` so
  repeated app runs/reruns don't re-tag the same reviews.
