"""
review_agent.py — an outbound, review-soliciting conversational agent.

Where `sephora_core.py` powers an *inbound* analyst chatbot ("what are people
complaining about?"), this module powers the reverse flow: proactively reaching
out to a customer a couple of weeks after they bought a product and running a
short, adaptive conversation to coax a useful review out of them.

The agent is *grounded in the review database*: before it says a word it looks
at how past buyers reviewed the same product (or, if that product is thin on
reviews, the same category) and turns the recurring themes into targeted
talking points. If lots of past reviewers said a cream broke them out, the
agent will make a point of asking whether it broke *this* buyer out too — and
it keeps steering the conversation until every anticipated theme is covered and
it has a clear read on overall sentiment.

Design notes
------------
* All control flow (when to reach out, which theme to probe next, when the
  answer is "satisfactory", when to stop) is deterministic and unit-testable,
  with no network access required.
* Natural phrasing of each message is delegated to the Groq LLM via
  `sephora_core.call_llm` *when* a GROQ_API_KEY is available; otherwise the
  agent falls back to friendly hand-written templates so it still works fully
  offline. Either way the conversation logic is identical.
* This module is framework-agnostic (no Streamlit imports) so it can be driven
  from the web UI, a CLI, a scheduler/cron job, or tests.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta

import pandas as pd

import sephora_core as core

# How long after purchase we wait before reaching out. Two weeks is long enough
# for the customer to actually form an opinion but recent enough to still
# remember the experience.
OUTREACH_MIN_DAYS = 14

# A product with fewer than this many reviews in the sample doesn't give us
# enough signal on its own, so we widen the lens to its whole category.
MIN_PRODUCT_REVIEWS = 3

# How many issue-themes we'll actively probe in one conversation. Beyond a
# handful it starts to feel like an interrogation.
MAX_TALKING_POINTS = 4

# Sentinel "issue" for the opening, open-ended impression question.
OVERALL = "__overall__"

# One targeted, natural question per issue label from sephora_core.ISSUE_LABELS.
# These are what the agent uses to zero in on themes past reviewers raised.
ISSUE_QUESTIONS = {
    "breakouts/acne": "Did you notice any breakouts, clogged pores, or new blemishes while using it?",
    "irritation/redness": "Did it cause any irritation, stinging, or redness on your skin?",
    "dryness/flaking": "Did your skin end up feeling dry, tight, or flaky after using it?",
    "oily/greasy": "Did it leave your skin feeling oily or greasy at all?",
    "bad smell": "How did the scent work for you — pleasant, or a bit off-putting?",
    "packaging issues": "How was the packaging — any trouble with the pump, cap, or leaking?",
    "too expensive": "Did it feel worth the price to you?",
    "no effect / didn't work": "Did you get the results you were hoping for, or did it not really do much?",
    "allergic reaction": "Did you have any kind of reaction to it, like itching or swelling?",
    "texture/consistency": "How did the texture feel — was it too sticky, thick, or runny?",
}

OVERALL_QUESTION = "To start, how are you liking it so far?"

# --- lightweight sentiment lexicon (used when no LLM key is set, and always to
# keep the deterministic "satisfaction" logic offline-friendly) -------------
_POSITIVE_WORDS = [
    "love", "loved", "loving", "great", "amazing", "wonderful", "excellent",
    "favorite", "favourite", "obsessed", "holy grail", "recommend", "glowing",
    "best", "perfect", "happy", "fantastic", "awesome", "impressed", "works",
    "worked", "gentle", "soft", "smooth", "hydrating", "life saver", "lifesaver",
]
_NEGATIVE_WORDS = [
    "hate", "hated", "terrible", "awful", "worst", "disappointed", "disappointing",
    "broke me out", "broke out", "breakout", "breakouts", "irritating", "irritated",
    "waste", "returned", "return it", "never again", "horrible", "bad", "useless",
    "dry", "greasy", "sticky", "burned", "burning", "stings", "regret",
]
_NEUTRAL_WORDS = [
    "okay", "ok", "fine", "alright", "meh", "decent", "average", "so-so", "so so",
    "mixed", "mediocre", "underwhelming",
]

# Signals the customer wants to bow out — respect it and stop.
_DECLINE_WORDS = [
    "no thanks", "not interested", "no thank you", "stop", "leave me alone",
    "don't want", "do not want", "not now", "maybe later", "unsubscribe", "busy",
]


def _word_pattern(words):
    escaped = [re.escape(w) for w in words]
    return re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)


_POSITIVE_RE = _word_pattern(_POSITIVE_WORDS)
_NEGATIVE_RE = _word_pattern(_NEGATIVE_WORDS)
_NEUTRAL_RE = _word_pattern(_NEUTRAL_WORDS)
_DECLINE_RE = _word_pattern(_DECLINE_WORDS)


# ----------------------------------------------------------------------
# OUTREACH TIMING
# ----------------------------------------------------------------------
def _as_date(value):
    """Coerce a date / datetime / ISO-ish string into a `date`."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse a date from {value!r}")
    return parsed.date()


def days_since_purchase(purchase_date, now=None):
    """Whole days elapsed between the purchase date and `now` (today by default)."""
    purchase = _as_date(purchase_date)
    now = _as_date(now) if now is not None else date.today()
    return (now - purchase).days


def is_due_for_outreach(purchase_date, now=None, min_days=OUTREACH_MIN_DAYS):
    """True once at least `min_days` have passed since purchase."""
    return days_since_purchase(purchase_date, now=now) >= min_days


def next_outreach_date(purchase_date, min_days=OUTREACH_MIN_DAYS):
    """The earliest date on which we should reach out about this purchase."""
    return _as_date(purchase_date) + timedelta(days=min_days)


# ----------------------------------------------------------------------
# GROUNDING: what did past buyers say about this product / category?
# ----------------------------------------------------------------------
def _match_product_reviews(sample, product_name=None, product_id=None):
    if product_id is not None and "product_id" in sample.columns:
        subset = sample[sample["product_id"].astype(str) == str(product_id)]
        if not subset.empty:
            return subset, "product"
    if product_name and "product_name" in sample.columns:
        subset = sample[sample["product_name"].astype(str).str.lower()
                        == str(product_name).lower()]
        if not subset.empty:
            return subset, "product"
    return sample.iloc[0:0], "none"


def product_review_insights(sample, product_name=None, product_id=None,
                            category=None, max_snippets=4):
    """
    Summarize how the database has reviewed this product so far.

    Returns a dict with the review count, average rating, the recurring issue
    themes (ordered most-common-first), a few representative snippets, and the
    scope actually used ("product" or "category") so callers can be transparent
    about how much of the signal is product-specific.
    """
    reviews, scope = _match_product_reviews(sample, product_name, product_id)

    # Fall back to the category when the product itself is barely reviewed.
    if len(reviews) < MIN_PRODUCT_REVIEWS:
        cat = category
        if cat is None and not reviews.empty and "category" in reviews.columns:
            cat = reviews["category"].iloc[0]
        if cat is not None and "category" in sample.columns:
            cat_reviews = sample[sample["category"].astype(str) == str(cat)]
            if len(cat_reviews) > len(reviews):
                reviews, scope = cat_reviews, "category"

    if reviews.empty:
        return {
            "scope": "none", "n_reviews": 0, "avg_rating": None,
            "issue_counts": Counter(), "top_issues": [],
            "snippets": [], "product_name": product_name, "category": category,
        }

    avg_rating = (round(float(reviews["rating"].mean()), 2)
                  if "rating" in reviews.columns else None)

    # Themes come from lower-rated reviews (the ones that flag problems), which
    # is exactly the material we want the agent to gently probe for.
    concern = reviews[reviews["rating"] <= 3] if "rating" in reviews.columns else reviews
    issue_counts = Counter()
    for text in concern["review_text"].astype(str):
        for issue in core.keyword_tag_issues(text):
            issue_counts[issue] += 1

    # A few short, representative snippets to ground the LLM's phrasing.
    snippets = []
    for text in reviews.sort_values("rating").get("review_text", pd.Series(dtype=str)).astype(str):
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break

    resolved_name = product_name
    if resolved_name is None and "product_name" in reviews.columns and scope == "product":
        resolved_name = reviews["product_name"].iloc[0]
    resolved_cat = category
    if resolved_cat is None and "category" in reviews.columns:
        resolved_cat = reviews["category"].iloc[0]

    return {
        "scope": scope,
        "n_reviews": int(len(reviews)),
        "avg_rating": avg_rating,
        "issue_counts": issue_counts,
        "top_issues": [issue for issue, _ in issue_counts.most_common()],
        "snippets": snippets,
        "product_name": resolved_name,
        "brand_name": (reviews["brand_name"].iloc[0]
                       if "brand_name" in reviews.columns and scope == "product" else None),
        "category": resolved_cat,
    }


def build_talking_points(insights, max_points=MAX_TALKING_POINTS):
    """
    Turn the insight themes into an ordered list of talking points the agent
    will work through: an opening impression question first, then the most
    common issue themes, each with a targeted question.
    """
    points = [{"issue": OVERALL, "question": OVERALL_QUESTION, "count": None}]
    for issue in insights.get("top_issues", []):
        if issue in ISSUE_QUESTIONS:
            points.append({
                "issue": issue,
                "question": ISSUE_QUESTIONS[issue],
                "count": insights["issue_counts"][issue],
            })
        if len(points) >= max_points + 1:  # +1 for the opening question
            break
    return points


# ----------------------------------------------------------------------
# SENTIMENT (deterministic, offline)
# ----------------------------------------------------------------------
def detect_sentiment(text):
    """Return 'positive' | 'negative' | 'neutral' | None for a single message."""
    text = str(text)
    pos = len(_POSITIVE_RE.findall(text))
    neg = len(_NEGATIVE_RE.findall(text))
    neu = len(_NEUTRAL_RE.findall(text))
    if pos == 0 and neg == 0 and neu == 0:
        return None
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    # Ties (including pure-neutral messages) read as neutral / mixed.
    return "neutral"


_SENTIMENT_TO_RATING = {"positive": 5, "neutral": 3, "negative": 2}


# ----------------------------------------------------------------------
# THE CONVERSATION
# ----------------------------------------------------------------------
class ReviewConversation:
    """
    Drives one review-soliciting conversation with a single customer.

    Typical loop::

        convo = ReviewConversation(insights)
        step = convo.advance()          # agent's opening message
        while not step["done"]:
            convo.ingest_user_message(user_reply)
            step = convo.advance()      # agent's next (adaptive) message

    `advance()` returns a dict: ``{"message": str, "done": bool, "draft": dict|None}``.
    """

    def __init__(self, insights, talking_points=None, llm_model="llama-3.1-8b-instant",
                 use_llm=True, max_points=MAX_TALKING_POINTS):
        self.insights = insights or {}
        self.talking_points = talking_points or build_talking_points(self.insights, max_points)
        self.llm_model = llm_model
        self.use_llm = use_llm

        self.history = []           # list of (role, text)
        self.answers = {}           # issue -> customer's answer text
        self.covered = set()        # issues considered addressed
        self.pending_issue = None   # issue we just asked about, awaiting a reply
        self.sentiment = None       # rolling overall read
        self._declined = False
        self._closed = False

    # -- state updates ------------------------------------------------
    def _update_sentiment(self, text):
        s = detect_sentiment(text)
        if s is None:
            return
        if self.sentiment is None or self.sentiment == "neutral":
            self.sentiment = s
        elif s != self.sentiment and s != "neutral":
            # Conflicting strong signals over the conversation → call it mixed.
            self.sentiment = "neutral"

    def _issue_labels(self):
        return {tp["issue"] for tp in self.talking_points}

    def ingest_user_message(self, text):
        """Record and interpret the customer's reply."""
        text = "" if text is None else str(text)
        self.history.append(("user", text))
        self._update_sentiment(text)

        if _DECLINE_RE.search(text):
            self._declined = True

        # The reply answers whatever we just asked about.
        if self.pending_issue is not None:
            self.answers[self.pending_issue] = text
            self.covered.add(self.pending_issue)
            self.pending_issue = None

        # Credit any themes the customer volunteered unprompted so we don't
        # redundantly ask about them later.
        labels = self._issue_labels()
        for issue in core.keyword_tag_issues(text):
            if issue in labels:
                self.covered.add(issue)
                self.answers.setdefault(issue, text)

    # -- satisfaction / stopping -------------------------------------
    def is_satisfied(self):
        """
        A response set is "satisfactory" once we have a clear overall read AND
        every anticipated theme has been addressed — or the customer opted out.
        """
        if self._declined:
            return True
        all_covered = all(tp["issue"] in self.covered for tp in self.talking_points)
        return self.sentiment is not None and all_covered

    # -- message generation ------------------------------------------
    def _next_talking_point(self):
        for tp in self.talking_points:
            if tp["issue"] not in self.covered:
                return tp
        return None

    def advance(self):
        """Produce the agent's next message (opening, follow-up, or close)."""
        if self._closed:
            return {"message": None, "done": True, "draft": self.review_draft()}

        if self._declined:
            self._closed = True
            msg = self._render_closing(declined=True)
            self.history.append(("assistant", msg))
            return {"message": msg, "done": True, "draft": None}

        tp = self._next_talking_point()
        if tp is None or self.is_satisfied():
            self._closed = True
            msg = self._render_closing(declined=False)
            self.history.append(("assistant", msg))
            return {"message": msg, "done": True, "draft": self.review_draft()}

        self.pending_issue = tp["issue"]
        msg = self._render_question(tp)
        self.history.append(("assistant", msg))
        return {"message": msg, "done": False, "draft": None}

    # -- rendering (LLM when available, templates otherwise) ---------
    def _product_label(self):
        return (self.insights.get("product_name")
                or self.insights.get("category")
                or "your recent purchase")

    def _render_question(self, tp):
        template = self._template_question(tp)
        if not (self.use_llm and self._has_llm_key()):
            return template
        return self._llm_question(tp, template)

    def _template_question(self, tp):
        product = self._product_label()
        if tp["issue"] == OVERALL:
            return (
                f"Hi! We noticed you picked up {product} about two weeks ago — "
                f"long enough to really put it to the test. We'd love to hear how "
                f"it's going and turn it into a quick review. {tp['question']}"
            )
        prefix = ""
        if self.sentiment == "negative":
            prefix = "Sorry to hear it hasn't been perfect. "
        elif self.sentiment == "positive":
            prefix = "Love that you're enjoying it! "
        # Ground the probe in the fact that other buyers flagged this.
        if tp.get("count"):
            prefix += "A few other reviewers mentioned this, so I'm curious: "
        return prefix + tp["question"]

    def _llm_question(self, tp, fallback):
        product = self._product_label()
        snippets = "\n".join(f"- {s}" for s in self.insights.get("snippets", [])[:3])
        history = core._format_history(self.history)
        if tp["issue"] == OVERALL:
            goal = ("Write a warm, brief opening message (2-3 sentences) reaching out to a "
                    "customer who bought this product ~2 weeks ago, asking for their overall "
                    "impression to kick off a review.")
        else:
            goal = (
                f"Write ONE short, friendly follow-up message that naturally works in this "
                f"specific question: \"{tp['question']}\". Other reviewers have raised this "
                f"theme, so it's worth checking. Acknowledge what the customer just said, then "
                f"ask."
            )
        prompt = (
            f"{history}"
            f"You are a friendly customer-experience agent collecting a product review for "
            f"'{product}'. Be conversational, concise, and never pushy.\n\n"
            f"What past reviewers said (for your context only, don't quote verbatim):\n"
            f"{snippets or '- (no examples available)'}\n\n"
            f"{goal}\n\nMessage:"
        )
        out = core.call_llm(prompt, model_name=self.llm_model)
        if not out or out.lower().startswith(("no groq", "could not reach")):
            return fallback
        return out.strip()

    def _render_closing(self, declined):
        if declined:
            return ("No problem at all — thanks for your time! If you change your mind, "
                    "you can leave a review anytime. Have a great day!")
        template = ("Thank you so much for the thoughtful feedback — this is really helpful! "
                    "Here's a quick draft of your review based on what you shared:\n\n"
                    + self._draft_text())
        if not (self.use_llm and self._has_llm_key()):
            return template
        return template  # keep the deterministic, transparent draft in the closing

    # -- structured output -------------------------------------------
    def review_draft(self):
        """A structured, submittable draft synthesized from the conversation."""
        highlights = [(issue, ans) for issue, ans in self.answers.items()]
        return {
            "product_name": self.insights.get("product_name"),
            "brand_name": self.insights.get("brand_name"),
            "category": self.insights.get("category"),
            "sentiment": self.sentiment,
            "suggested_rating": _SENTIMENT_TO_RATING.get(self.sentiment),
            "declined": self._declined,
            "highlights": highlights,
            "text": self._draft_text(),
        }

    def _draft_text(self):
        if self._declined:
            return "(Customer declined to leave a review.)"
        parts = []
        overall = self.answers.get(OVERALL)
        if overall:
            parts.append(overall.strip())
        for tp in self.talking_points:
            issue = tp["issue"]
            if issue == OVERALL:
                continue
            ans = self.answers.get(issue)
            if ans:
                label = issue.split("/")[0].strip().capitalize()
                parts.append(f"{label}: {ans.strip()}")
        if not parts:
            return "(No review content collected yet.)"
        return "\n".join(parts)

    @staticmethod
    def _has_llm_key():
        import os
        return bool(os.environ.get("GROQ_API_KEY"))


# ----------------------------------------------------------------------
# CLI DEMO (scripted, so it runs without a live human or an API key)
# ----------------------------------------------------------------------
def _demo():
    import os

    sample = pd.read_parquet(core.SAMPLE_PATH)
    product = sample["product_name"].dropna().iloc[0]
    insights = product_review_insights(sample, product_name=product)

    print(f"Product: {insights['product_name']}  "
          f"(scope={insights['scope']}, n={insights['n_reviews']}, "
          f"avg_rating={insights['avg_rating']})")
    print(f"Anticipated themes: {insights['top_issues'] or '(none)'}\n")

    convo = ReviewConversation(insights, use_llm=bool(os.environ.get("GROQ_API_KEY")))

    # Canned customer replies for a repeatable, network-free demo.
    scripted = iter([
        "Honestly I love it so far, my skin looks great!",
        "No breakouts at all, actually cleared me up a bit.",
        "No irritation either, super gentle.",
        "It is a little pricey but I think it's worth it.",
    ])

    step = convo.advance()
    while not step["done"]:
        print(f"AGENT:    {step['message']}")
        try:
            reply = next(scripted)
        except StopIteration:
            reply = "That's about it, thanks!"
        print(f"CUSTOMER: {reply}\n")
        convo.ingest_user_message(reply)
        step = convo.advance()

    print(f"AGENT:    {step['message']}\n")
    print("STRUCTURED DRAFT:")
    draft = step["draft"] or convo.review_draft()
    print(f"  suggested_rating: {draft['suggested_rating']}  sentiment: {draft['sentiment']}")


if __name__ == "__main__":
    _demo()
