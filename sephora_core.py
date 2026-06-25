"""
sephora_core.py — shared logic for the stats + RAG pipeline.
Used by both sephora_pipeline.py (CLI) and app.py (Streamlit web UI).
"""

import glob
import json
import os
from collections import Counter

import pandas as pd

SAMPLE_PATH = "sample_10k.parquet"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sephora_reviews"
SAMPLE_SIZE = 1000
PER_GROUP_CAP = 250

# Keyword tagger kept as a free, instant fallback if the Groq API isn't available
ISSUE_KEYWORDS = {
    "breakouts/acne": ["breakout", "acne", "pimple", "blemish", "clogg"],
    "irritation/redness": ["irritat", "burn", "sting", "redness", "rash", "inflam"],
    "dryness/flaking": ["dry", "flak", "tight", "dehydrat"],
    "oily/greasy": ["oily", "greasy", "shine", "shiny"],
    "bad smell": ["smell", "scent", "odor", "stink"],
    "packaging issues": ["leak", "broke", "broken", "pump", "packaging", "spill", "cap"],
    "too expensive": ["expensive", "overpriced", "price", "pricey", "cost"],
    "no effect / didn't work": ["didn't work", "did not work", "no effect", "no difference",
                                 "waste of money", "didn't help", "useless"],
    "allergic reaction": ["allerg", "reaction", "hives", "swell"],
    "texture/consistency": ["greasy", "sticky", "gritty", "watery", "thick", "runny"],
}

ISSUE_LABELS = list(ISSUE_KEYWORDS.keys())

MONTH_NAMES = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
               7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}
MONTH_LOOKUP = {v: k for k, v in MONTH_NAMES.items()}

STAT_TRIGGERS = ["most common", "common issue", "frequent", "how many", "trend",
                  "expected", "prone to", "stats", "statistic", "breakdown", "compare"]


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

    cat_col = next((c for c in ["secondary_category"] if c in df.columns), None)
    if cat_col is None:
        df["secondary_category"] = "unknown"
    else:
        df = df.rename(columns={cat_col: "secondary_category"})
        df["secondary_category"] = df["secondary_category"].fillna("unknown")

    
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
    print(df.columns.tolist())


    # Safety check: remove accidental duplicate columns
    #df = df.loc[:, ~df.columns.duplicated()]
    grouped = df.groupby(["secondary_category", "rating_bucket"], group_keys=False)
    idx_per_group = grouped.apply(
        lambda x: x.sample(min(len(x), per_group_cap), random_state=seed).index,
        include_groups=False,
    )
    import numpy as np
    all_idx = pd.Index(np.concatenate(idx_per_group.values)) if len(idx_per_group) else pd.Index([])
    sample = df.loc[all_idx]

    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=seed)

    keep_cols = [c for c in ["review_text", "rating", "secondary_category", "month",
                              "product_id", "product_name", "brand_name"] if c in sample.columns]
    sample = sample[keep_cols].reset_index(drop=True)
    sample = sample.rename(columns={"secondary_category": "primary_category"})
    return sample


# ----------------------------------------------------------------------
# ISSUE TAGGING: keyword (fast/free) or LLM-based (slower, more accurate)
# ----------------------------------------------------------------------
def keyword_tag_issues(text):
    text = str(text).lower()
    return [issue for issue, kws in ISSUE_KEYWORDS.items() if any(k in text for k in kws)]


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
            # fallback for this batch: keyword tagging
            for i, text in enumerate(batch):
                results[start + i] = keyword_tag_issues(text)

        if progress_cb:
            progress_cb(min(start + batch_size, len(texts)), len(texts))

    return results


# ----------------------------------------------------------------------
# AGGREGATES
# ----------------------------------------------------------------------
def build_aggregates(sample, use_llm_tagger=False, llm_model="llama-3.1-8b-instant", progress_cb=None):
    neg = sample[sample["rating"] <= 2].copy()

    if use_llm_tagger:
        neg["issues"] = llm_tag_issues_batch(neg["review_text"].astype(str).tolist(),
                                              model_name=llm_model, progress_cb=progress_cb)
    else:
        neg["issues"] = neg["review_text"].apply(keyword_tag_issues)

    exploded = neg.explode("issues").dropna(subset=["issues"])

    overall = Counter([i for issues in neg["issues"] for i in issues])

    by_category = (exploded.groupby(["primary_category", "issues"])
                            .size().reset_index(name="count")
                            .sort_values("count", ascending=False))

    by_month = (exploded.dropna(subset=["month"])
                         .groupby(["month", "issues"])
                         .size().reset_index(name="count")
                         .sort_values("count", ascending=False)) if "month" in exploded.columns else pd.DataFrame()

    by_month_category = (exploded.dropna(subset=["month"])
                                  .groupby(["month", "primary_category", "issues"])
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

    metas = sample[[c for c in ["primary_category", "month", "rating", "product_name"]
                    if c in sample.columns]].fillna("").astype(str).to_dict("records")

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
def is_stat_question(q):
    q = q.lower()
    return any(t in q for t in STAT_TRIGGERS)


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
                                         (aggs["by_month_category"]["primary_category"] == category)]
        lines.append(f"Issue counts for category='{category}' in month={month} ({MONTH_NAMES[month]}):")
        lines.append(sub.head(10).to_string(index=False))
    elif month and not aggs["by_month"].empty:
        sub = aggs["by_month"][aggs["by_month"]["month"] == month]
        lines.append(f"Issue counts across all categories in month={month} ({MONTH_NAMES[month]}):")
        lines.append(sub.head(10).to_string(index=False))
    elif category:
        sub = aggs["by_category"][aggs["by_category"]["primary_category"] == category]
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
        return ("[No GROQ_API_KEY found. Set it as an environment variable, or as a "
                "Streamlit secret if deployed, then try again.]\n\n"
                f"Here is the raw context that would have been sent to the LLM:\n\n{prompt}")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return (f"[Could not reach Groq API: {e}]\n\n"
                f"Here is the raw context that would have been sent to the LLM:\n\n{prompt}")


def answer(question, aggs, categories, model=None, collection=None, llm_model="llama-3.1-8b-instant"):
    if is_stat_question(question):
        context = build_stat_context(question, aggs, categories)
        prompt = (
            "You are a product-feedback analyst. Answer the question ONLY using the data below. "
            "Be specific and cite numbers. If the data doesn't cover the question, say so.\n\n"
            f"DATA:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
    else:
        if model is None or collection is None:
            return "Vector index not loaded — can't answer open-ended questions right now."
        q_emb = model.encode([question]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=8)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        context = "\n---\n".join(
            f"[{m.get('primary_category', '')}, rating={m.get('rating', '')}] {d}"
            for d, m in zip(docs, metas)
        )
        prompt = (
            "You are a product-feedback analyst. Answer the question using only the example reviews below.\n\n"
            f"REVIEWS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
    return call_llm(prompt, model_name=llm_model)
