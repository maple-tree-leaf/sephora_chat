# Sephora Skincare Reviews — Stats + RAG Chatbot + Review-Solicitation Agent

A Streamlit app with three modes built on top of Sephora skincare product
reviews:

- **Stat-style questions** ("What are the most common complaints?", "What's
  most common in June for Moisturizers?") are answered from precomputed issue
  counts, tagged either with fast keyword matching or via the Groq API.
- **Open-ended questions** ("Why do people say this didn't work for them?")
  are answered with retrieval-augmented generation (RAG) over a semantic
  search index of a cached review sample.
- **Review outreach** — a proactive agent that, ~2 weeks after a customer buys
  a product, reaches out and runs a short, adaptive conversation to collect a
  review. Crucially, it's *grounded in the review database*: it reads how past
  buyers reviewed the same product, turns recurring themes into targeted
  questions (e.g. if lots of past reviews mention breakouts, it asks whether
  this buyer broke out too), and keeps steering the chat until it has covered
  every anticipated theme and has a clear read on overall sentiment.

## The review-solicitation agent

The agent lives in `review_agent.py` and is framework-agnostic, so it can be
driven from the Streamlit **📣 Review outreach** tab, a scheduler/cron job, or
a script. Its flow:

1. **Timing** — `is_due_for_outreach(purchase_date)` returns `True` once
   `OUTREACH_MIN_DAYS` (14) have passed since purchase.
2. **Grounding** — `product_review_insights(sample, product_name=...)` pulls
   the product's past reviews (falling back to the category when the product
   itself is thin on reviews), computes the average rating, and ranks the
   recurring issue themes using the same keyword tagger as the stats pipeline.
3. **Talking points** — `build_talking_points(insights)` turns those themes
   into an ordered agenda: an open impression question first, then a targeted
   question per top theme.
4. **Adaptive conversation** — `ReviewConversation` runs the exchange. Each
   customer reply updates a rolling sentiment read and marks themes as covered
   (including ones the customer volunteers unprompted). The agent probes the
   next uncovered theme, adapts its tone to sentiment, and stops once every
   theme is covered and sentiment is known — or the customer declines.
5. **Output** — on completion it emits a structured `review_draft()` with a
   suggested star rating, sentiment, per-theme highlights, and draft review
   text.

All control flow (timing, theme selection, satisfaction, stopping) is
deterministic and offline-testable. Natural message phrasing is delegated to
the Groq LLM when a `GROQ_API_KEY` is set, and falls back to friendly built-in
templates otherwise, so the agent works fully offline.

Try the scripted, network-free demo:

```bash
python review_agent.py
```

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: sidebar setup, stats dashboard, analyst chat, and the review-outreach agent tab. |
| `sephora_core.py` | Shared logic: sampling, issue tagging, aggregates, the vector index, and question routing/answering. Framework-agnostic so it's easy to test or reuse from a script/CLI. |
| `review_agent.py` | The outbound review-solicitation agent: outreach timing, product/category grounding, adaptive conversation, and structured review drafts. Also runs a scripted demo via `python review_agent.py`. |
| `sample_1k.parquet` | A cached 1,000-review sample (already tagged-ready), committed to the repo so the app runs out of the box without the full ~1M-row raw dataset. |
| `tests/test_core.py` | Unit tests for the pure logic in `sephora_core.py` (no network/API calls). |
| `tests/test_review_agent.py` | Unit tests for `review_agent.py`: timing, grounding, sentiment, and the full conversation flow (no network/API calls). |

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
in `review_agent.py` (outreach timing, product/category grounding, sentiment
detection, and the conversation control flow). They run without a
`GROQ_API_KEY` or network access.

## Notes on issue tagging

- **Keyword tagging** (default) is instant and free but relies on a curated
  keyword/stem list (`ISSUE_KEYWORDS` in `sephora_core.py`). Words are matched
  as whole words or word-stems, so e.g. `"cap"` doesn't fire on `"capacity"`.
- **LLM-based tagging** (toggle in the sidebar) sends batches of negative
  reviews to the Groq API for more accurate, phrasing-agnostic tagging. It's
  slower and uses more API calls, but results are cached under `.cache/` so
  repeated app runs/reruns don't re-tag the same reviews.
