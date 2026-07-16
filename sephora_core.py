"""
sephora_core.py — shared logic for the stats + RAG pipeline.
Used by app.py (the Streamlit web UI).
"""

import glob
import json
import logging
import os
import re
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_PATH = "sample_1k.parquet"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sephora_reviews"
CACHE_DIR = ".cache"
SAMPLE_SIZE = 1000
PER_GROUP_CAP = 250

# Keyword tagger kept as a free, instant fallback if the Groq API isn't available.
# Entries that end in "\w*" are deliberately truncated stems (e.g. "dry\w*" also
# matches "dried"/"drying"); everything else is matched as a whole word only, so
# short/common substrings like "cap" or "cost" don't fire on unrelated words like
# "capacity" or "costume".
ISSUE_KEYWORDS = {
    # "broke\s+(?:me\s+)?out" catches the very common "broke me out" phrasing
    # before the "packaging issues" pattern below gets a chance to claim the
    # bare word "broke" for a broken bottle/pump instead.
    "breakouts/acne": ["breakout", r"broke\s+(?:me\s+)?out", "acne", "pimple", "blemish", r"clogg\w*"],
    "irritation/redness": [r"irritat\w*", r"burn(?:ing|s|ed)?", r"sting\w*",
                            "redness", "rash", r"inflam\w*"],
    "dryness/flaking": [r"dry\w*", r"flak\w*", "tight", r"dehydrat\w*"],
    "oily/greasy": ["oily", "greasy", r"shin(?:e|y|ing)"],
    "bad smell": [r"smell\w*", r"scent\w*", "odor", r"stink\w*"],
    "packaging issues": [r"leak\w*", r"broke(?!\s+(?:me\s+)?out)", "broken", "pump", "packaging",
                          r"spill\w*", "cap"],
    "too expensive": ["expensive", "overpriced", "price", "pricey", "cost"],
    "no effect / didn't work": [r"didn.?t work", "did not work", "no effect",
                                 "no difference", "waste of money", r"didn.?t help",
                                 "useless"],
    "allergic reaction": [r"allerg\w*", "reaction", "hives", r"swell\w*", "swollen"],
    "texture/consistency": ["sticky", "gritty", "watery", r"thick\w*", "runny"],
}

ISSUE_LABELS = list(ISSUE_KEYWORDS.keys())

# Precompiled per-issue regex: each keyword/stem is wrapped in word boundaries so
# e.g. "cap" only matches the standalone word "cap", never "capacity"/"escape",
# while "dry\w*" still matches "dry", "dried", "drying", etc.
ISSUE_PATTERNS = {
    issue: re.compile(r"\b(?:" + "|".join(kws) + r")\b", re.IGNORECASE)
    for issue, kws in ISSUE_KEYWORDS.items()
}

MONTH_NAMES = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
               7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}
MONTH_LOOKUP = {v: k for k, v in MONTH_NAMES.items()}

STAT_TRIGGER_PHRASES = ["most common", "common issue", "frequent", "how many", "trend",
                         "expected", "prone to", "stats", "statistic", "breakdown", "compare"]
STAT_TRIGGER_PATTERNS = [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
                          for p in STAT_TRIGGER_PHRASES]
# If a question mentions a known category/month AND one of these generic
# "issue" words, route it to stats even without an exact trigger phrase.
ISSUE_HINT_WORDS = ["issue", "problem", "complain", "complaint"]
ISSUE_HINT_PATTERN = re.compile(r"\b(?:" + "|".join(ISSUE_HINT_WORDS) + r")\w*\b", re.IGNORECASE)


# ----------------------------------------------------------------------
# LOAD + SAMPLE
# ----------------------------------------------------------------------
def load_raw(data_dir):
    product_path = os.path.join(data_dir, "product_info.csv")
    if not os.path.exists(product_path):
        raise FileNotFoundError(f"Couldn't find product_info.csv in {data_dir}")
    products = pd.read_csv(product_path)

    review_files = sorted(glob.glob(os.path.join(data_dir, "reviews*.csv")))
    if not review_files:
        raise FileNotFoundError(f"Couldn't find any reviews*.csv files in {data_dir}")

    reviews = pd.concat([pd.read_csv(f) for f in review_files], ignore_index=True)
    df = reviews.merge(products, on="product_id", how="left", suffixes=("", "_prod"))
    return df, len(reviews), len(products)


def build_sample(df, sample_size=SAMPLE_SIZE, per_group_cap=PER_GROUP_CAP, seed=42):
    df = df.copy()

    text_col = next((c for c in ["review_text", "text", "review"] if c in df.columns), None)
    if text_col is None:
        raise KeyError("No review text column found.")
    df = df.rename(columns={text_col: "review_text"})
    df = df[df["review_text"].notna()]

    date_col = next((c for c in ["submission_time", "review_date", "date"] if c in df.columns), None)
    if date_col:
        df["submission_time"] = pd.to_datetime(df[date_col], errors="coerce")
        df["month"] = df["submission_time"].dt.month
    else:
        df["month"] = None

    # Sephora's raw data only has one usable category field for our purposes
    # (secondary_category); we call it "category" downstream to avoid implying
    # it's the same as any "primary_category" field the raw dataset may have.
    cat_col = next((c for c in ["secondary_category"] if c in df.columns), None)
    if cat_col is None:
        df["category"] = "unknown"
    else:
        df = df.rename(columns={cat_col: "category"})
        df["category"] = df["category"].fillna("unknown")

    rating_col = next((c for c in ["rating", "review_rating", "star_rating"] if c in df.columns), None)
    if rating_col is None:
        raise KeyError("No rating column found.")
    df = df.rename(columns={rating_col: "rating"})

    df["rating_bucket"] = pd.cut(df["rating"], [0, 2, 3, 5],
                                  labels=["negative", "neutral", "positive"])

    # NOTE: pandas >=2.2 drops the grouping columns from groupby().apply() results
    # (and pandas 3.x forbids overriding this), so we sample indices per group
    # instead of relying on the grouping columns surviving inside apply().
    df = df.dropna(subset=["rating_bucket"]).reset_index(drop=True)

    grouped = df.groupby(["category", "rating_bucket"], group_keys=False)
    idx_per_group = grouped.apply(
        lambda x: x.sample(min(len(x), per_group_cap), random_state=seed).index,
        include_groups=False,
    )
    import numpy as np
    all_idx = pd.Index(np.concatenate(idx_per_group.values)) if len(idx_per_group) else pd.Index([])
    sample = df.loc[all_idx]

    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=seed)

    keep_cols = [c for c in ["review_text", "rating", "category", "month",
                              "product_id", "product_name", "brand_name"] if c in sample.columns]
    sample = sample[keep_cols].reset_index(drop=True)
    return sample


# ----------------------------------------------------------------------
# ISSUE TAGGING: keyword (fast/free) or LLM-based (slower, more accurate)
# ----------------------------------------------------------------------
def keyword_tag_issues(text):
    text = str(text)
    return [issue for issue, pattern in ISSUE_PATTERNS.items() if pattern.search(text)]


def llm_tag_issues_batch(texts, model_name="llama-3.1-8b-instant", batch_size=20, progress_cb=None):
    """
    Tags a list of review texts with issue labels using the Groq API.
    Sends small batches as a JSON-producing prompt to keep call count low.
    Falls back to keyword tagging per-item if the model output can't be parsed.
    Returns a list of lists (same order as input texts).
    """
    label_list = ", ".join(ISSUE_LABELS)
    results = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        numbered = "\n".join(f"{i}: {t[:500]}" for i, t in enumerate(batch))
        prompt = (
            f"You are tagging negative product reviews with issue categories.\n"
            f"Allowed categories (use these exact strings only): {label_list}\n\n"
            f"For each numbered review below, return the matching categories (zero or more per review).\n"
            f"Respond ONLY with valid JSON: a dict mapping the review number (as a string) to a list of category strings. "
            f"No extra text, no markdown fences.\n\n"
            f"Reviews:\n{numbered}\n\nJSON:"
        )
        try:
            raw = call_llm(prompt, model_name=model_name).strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            for i, text in enumerate(batch):
                tags = parsed.get(str(i), [])
                tags = [t for t in tags if t in ISSUE_LABELS]
                results[start + i] = tags if tags else keyword_tag_issues(text)
        except Exception:
            logger.warning("LLM issue-tagging batch failed; falling back to keyword tagging.", exc_info=True)
            for i, text in enumerate(batch):
                results[start + i] = keyword_tag_issues(text)

        if progress_cb:
            progress_cb(min(start + batch_size, len(texts)), len(texts))

    return results


# ----------------------------------------------------------------------
# ISSUE-TAG CACHING (avoid re-tagging identical data across app reruns)
# ----------------------------------------------------------------------
def _issue_cache_path(tagger_key):
    return os.path.join(CACHE_DIR, f"issues_{tagger_key}.json")


def _load_issue_cache(tagger_key):
    path = _issue_cache_path(tagger_key)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not read issue cache at %s; ignoring it.", path, exc_info=True)
        return {}


def _save_issue_cache(tagger_key, cache):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_issue_cache_path(tagger_key), "w") as f:
            json.dump(cache, f)
    except Exception:
        logger.warning("Could not write issue cache for %s.", tagger_key, exc_info=True)


# ----------------------------------------------------------------------
# AGGREGATES
# ----------------------------------------------------------------------
def build_aggregates(sample, use_llm_tagger=False, llm_model="llama-3.1-8b-instant",
                      progress_cb=None, force_retag=False):
    neg = sample[sample["rating"] <= 2].copy()
    idx_list = neg.index.tolist()
    texts = neg["review_text"].astype(str).tolist()

    tagger_key = "keyword" if not use_llm_tagger else f"llm_{llm_model}"
    cache = {} if force_retag else _load_issue_cache(tagger_key)

    if use_llm_tagger:
        missing_positions = [i for i, idx in enumerate(idx_list) if str(idx) not in cache]
        if missing_positions:
            missing_texts = [texts[i] for i in missing_positions]
            tagged = llm_tag_issues_batch(missing_texts, model_name=llm_model, progress_cb=progress_cb)
            for pos, tags in zip(missing_positions, tagged):
                cache[str(idx_list[pos])] = tags
            _save_issue_cache(tagger_key, cache)
        elif progress_cb:
            progress_cb(len(idx_list), len(idx_list))
        neg["issues"] = [cache[str(idx)] for idx in idx_list]
    else:
        neg["issues"] = [keyword_tag_issues(t) for t in texts]
        if progress_cb:
            progress_cb(len(idx_list), len(idx_list))
        # Keyword tagging is cheap, but caching it too keeps behavior consistent
        # with the LLM path and makes debugging tag assignments easier.
        _save_issue_cache(tagger_key, {str(idx): tags for idx, tags in zip(idx_list, neg["issues"])})

    exploded = neg.explode("issues").dropna(subset=["issues"])

    overall = Counter([i for issues in neg["issues"] for i in issues])

    by_category = (exploded.groupby(["category", "issues"])
                            .size().reset_index(name="count")
                            .sort_values("count", ascending=False))

    by_month = (exploded.dropna(subset=["month"])
                         .groupby(["month", "issues"])
                         .size().reset_index(name="count")
                         .sort_values("count", ascending=False)) if "month" in exploded.columns else pd.DataFrame()

    by_month_category = (exploded.dropna(subset=["month"])
                                  .groupby(["month", "category", "issues"])
                                  .size().reset_index(name="count")
                                  .sort_values("count", ascending=False)) if "month" in exploded.columns else pd.DataFrame()

    return {
        "overall": overall,
        "by_category": by_category,
        "by_month": by_month,
        "by_month_category": by_month_category,
    }


# ----------------------------------------------------------------------
# VECTOR INDEX
# ----------------------------------------------------------------------
def _format_month_meta(m):
    try:
        return str(int(m)) if pd.notna(m) else ""
    except (TypeError, ValueError):
        return ""


def build_index(sample, persist_dir=CHROMA_DIR):
    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = sample["review_text"].astype(str).tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    meta_cols = [c for c in ["category", "month", "rating", "product_name"] if c in sample.columns]
    meta_df = sample[meta_cols].copy()
    if "month" in meta_df.columns:
        meta_df["month"] = meta_df["month"].apply(_format_month_meta)
    metas = meta_df.fillna("").astype(str).to_dict("records")

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metas,
        ids=[str(i) for i in range(len(texts))],
    )
    return model, collection


def load_index(persist_dir=CHROMA_DIR):
    from sentence_transformers import SentenceTransformer
    import chromadb
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


# ----------------------------------------------------------------------
# ROUTING + ANSWERING
# ----------------------------------------------------------------------
def is_stat_question(q, categories=None):
    if any(p.search(q) for p in STAT_TRIGGER_PATTERNS):
        return True
    if categories and ISSUE_HINT_PATTERN.search(q) and extract_category(q, categories):
        return True
    return False


def extract_month(q):
    q = q.lower()
    for name, num in MONTH_LOOKUP.items():
        if name in q:
            return num
    return None


def extract_category(q, categories):
    q = q.lower()
    for cat in categories:
        if str(cat).lower() in q:
            return cat
    return None


def build_stat_context(q, aggs, categories):
    month = extract_month(q)
    category = extract_category(q, categories)
    lines = []

    if month and category and not aggs["by_month_category"].empty:
        sub = aggs["by_month_category"][(aggs["by_month_category"]["month"] == month) &
                                         (aggs["by_month_category"]["category"] == category)]
        lines.append(f"Issue counts for category='{category}' in month={month} ({MONTH_NAMES[month]}):")
        lines.append(sub.head(10).to_string(index=False))
    elif month and not aggs["by_month"].empty:
        sub = aggs["by_month"][aggs["by_month"]["month"] == month]
        lines.append(f"Issue counts across all categories in month={month} ({MONTH_NAMES[month]}):")
        lines.append(sub.head(10).to_string(index=False))
    elif category:
        sub = aggs["by_category"][aggs["by_category"]["category"] == category]
        lines.append(f"Issue counts for category='{category}':")
        lines.append(sub.head(10).to_string(index=False))
    else:
        lines.append("Overall most common issues across all negative reviews:")
        lines.append("\n".join(f"{k}: {v}" for k, v in aggs["overall"].most_common(10)))
        lines.append("\nTop issues broken down by category:")
        lines.append(aggs["by_category"].head(15).to_string(index=False))

    return "\n".join(str(l) for l in lines)


def call_llm(prompt, model_name="llama-3.1-8b-instant"):
    """
    Calls Groq's hosted API (free tier, OpenAI-compatible) instead of a local
    Ollama model — this is what lets the app run on a cloud host like
    Streamlit Community Cloud, where a local model server can't run.

    Requires a GROQ_API_KEY environment variable (set as a Streamlit "secret"
    when deployed, or a local env var when running on your own machine).
    Get a free key at https://console.groq.com/keys
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.info("No GROQ_API_KEY set; returning a fallback message instead of calling Groq.")
        return ("No GROQ_API_KEY found. Set it as an environment variable, or as a "
                "Streamlit secret if deployed, then try again.")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("Groq API call failed. Prompt (truncated): %s", prompt[:500], exc_info=True)
        return f"Could not reach the Groq API right now ({e}). Please try again in a moment."


def _format_history(history, max_turns=3):
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines = [f"{role}: {content}" for role, content in recent]
    return "Conversation so far (for context on follow-up questions):\n" + "\n".join(lines) + "\n\n"


def _build_rag_where(month, category):
    conditions = []
    if category:
        conditions.append({"category": str(category)})
    if month:
        conditions.append({"month": str(month)})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def answer(question, aggs, categories, model=None, collection=None,
           llm_model="llama-3.1-8b-instant", history=None):
    """Returns (answer_text, sources). `sources` is a list of dicts (empty for
    stat-style answers, populated with the retrieved reviews for RAG answers)."""
    history_block = _format_history(history)

    if is_stat_question(question, categories):
        context = build_stat_context(question, aggs, categories)
        prompt = (
            f"{history_block}"
            "You are a product-feedback analyst. Answer the question ONLY using the data below. "
            "Be specific and cite numbers. If the data doesn't cover the question, say so.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        return call_llm(prompt, model_name=llm_model), []

    if model is None or collection is None:
        return "Vector index not loaded — can't answer open-ended questions right now.", []

    month = extract_month(question)
    category = extract_category(question, categories)
    where = _build_rag_where(month, category)

    q_emb = model.encode([question]).tolist()
    query_kwargs = {"query_embeddings": q_emb, "n_results": 8}
    if where:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if where and not docs:
        # The filter matched nothing (e.g. no reviews for that category+month
        # combo in the sample) — fall back to an unfiltered search rather than
        # telling the user there's no data at all.
        results = collection.query(query_embeddings=q_emb, n_results=8)
        docs = results["documents"][0]
        metas = results["metadatas"][0]

    sources = [
        {
            "category": m.get("category", ""),
            "month": m.get("month", ""),
            "rating": m.get("rating", ""),
            "product_name": m.get("product_name", ""),
            "text": d,
        }
        for d, m in zip(docs, metas)
    ]

    context = "\n---\n".join(
        f"[{s['category']}, rating={s['rating']}] {s['text']}" for s in sources
    )
    prompt = (
        f"{history_block}"
        "You are a product-feedback analyst. Answer the question using only the example reviews below.\n\n"
        f"REVIEWS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    return call_llm(prompt, model_name=llm_model), sources
