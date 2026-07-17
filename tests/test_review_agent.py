"""
Unit tests for review_agent.py — the outbound review-solicitation agent.

These cover the deterministic, offline logic only: outreach timing, grounding
insights, talking-point construction, sentiment detection, and the full
conversation control flow (adaptive probing, satisfaction, decline, and the
structured draft). They never touch the Groq API or any network, so they run
fast and without a GROQ_API_KEY.
"""

import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Guarantee the offline (template) path regardless of the runner's environment.
os.environ.pop("GROQ_API_KEY", None)

import review_agent as ra


# ----------------------------------------------------------------------
# outreach timing
# ----------------------------------------------------------------------
def test_days_since_purchase_counts_whole_days():
    assert ra.days_since_purchase("2024-01-01", now="2024-01-15") == 14


def test_days_since_purchase_accepts_date_and_datetime():
    assert ra.days_since_purchase(date(2024, 1, 1), now=datetime(2024, 1, 8)) == 7


def test_is_due_for_outreach_true_at_threshold():
    assert ra.is_due_for_outreach("2024-01-01", now="2024-01-15") is True


def test_is_due_for_outreach_false_before_threshold():
    assert ra.is_due_for_outreach("2024-01-01", now="2024-01-10") is False


def test_next_outreach_date_is_two_weeks_out():
    assert ra.next_outreach_date("2024-01-01") == date(2024, 1, 15)


# ----------------------------------------------------------------------
# grounding: product_review_insights
# ----------------------------------------------------------------------
def _sample_df():
    return pd.DataFrame({
        "review_text": [
            "This cream totally broke me out, so many pimples.",
            "Broke out along my jaw, acne everywhere.",
            "Smells awful and gave me a breakout too.",
            "Absolutely love it, cleared my skin!",
            "Great product, no complaints.",
        ],
        "rating": [1, 2, 1, 5, 5],
        "category": ["Moisturizers"] * 5,
        "product_id": ["p1"] * 5,
        "product_name": ["Miracle Cream"] * 5,
        "brand_name": ["BrandX"] * 5,
        "month": [6, 6, 7, 7, 8],
    })


def test_product_insights_surfaces_dominant_theme():
    insights = ra.product_review_insights(_sample_df(), product_name="Miracle Cream")
    assert insights["scope"] == "product"
    assert insights["n_reviews"] == 5
    assert insights["top_issues"][0] == "breakouts/acne"
    assert insights["issue_counts"]["breakouts/acne"] == 3


def test_product_insights_falls_back_to_category_when_product_thin():
    df = _sample_df()
    # A product with a single review should borrow signal from its category.
    df.loc[df.index[-1], "product_id"] = "p2"
    df.loc[df.index[-1], "product_name"] = "Rare Serum"
    insights = ra.product_review_insights(df, product_name="Rare Serum")
    assert insights["scope"] == "category"
    assert insights["n_reviews"] == 5


def test_product_insights_empty_when_no_match_and_no_category():
    df = _sample_df().drop(columns=["category"])
    insights = ra.product_review_insights(df, product_name="Nonexistent")
    assert insights["n_reviews"] == 0
    assert insights["top_issues"] == []


# ----------------------------------------------------------------------
# talking points
# ----------------------------------------------------------------------
def test_build_talking_points_starts_with_overall_then_issues():
    insights = ra.product_review_insights(_sample_df(), product_name="Miracle Cream")
    tps = ra.build_talking_points(insights)
    assert tps[0]["issue"] == ra.OVERALL
    assert any(tp["issue"] == "breakouts/acne" for tp in tps)


def test_build_talking_points_respects_max():
    insights = {
        "top_issues": list(ra.ISSUE_QUESTIONS.keys()),
        "issue_counts": {k: 1 for k in ra.ISSUE_QUESTIONS},
    }
    tps = ra.build_talking_points(insights, max_points=2)
    # opening + at most 2 issue points
    assert len(tps) == 3


# ----------------------------------------------------------------------
# sentiment
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("I absolutely love this, works great!", "positive"),
    ("It broke me out and I hate it", "negative"),
    ("It's okay, kind of average", "neutral"),
    ("The bottle is blue", None),
])
def test_detect_sentiment(text, expected):
    assert ra.detect_sentiment(text) == expected


# ----------------------------------------------------------------------
# conversation flow
# ----------------------------------------------------------------------
def _convo():
    insights = ra.product_review_insights(_sample_df(), product_name="Miracle Cream")
    return ra.ReviewConversation(insights, use_llm=False)


def test_conversation_opens_with_overall_question():
    convo = _convo()
    step = convo.advance()
    assert step["done"] is False
    assert convo.pending_issue == ra.OVERALL
    assert "Miracle Cream" in step["message"]


def test_conversation_probes_anticipated_theme():
    convo = _convo()
    convo.advance()  # opening
    convo.ingest_user_message("I really love it so far!")
    step = convo.advance()
    # The dominant theme (breakouts) should be probed.
    assert convo.pending_issue == "breakouts/acne"
    assert step["done"] is False


def test_conversation_reaches_satisfaction_and_emits_draft():
    convo = _convo()
    step = convo.advance()
    replies = [
        "I love it so far, skin looks amazing!",
        "No breakouts at all, actually improved my skin.",
        "The scent is lovely too.",
        "Worth every penny.",
    ]
    i = 0
    while not step["done"] and i < 10:
        convo.ingest_user_message(replies[min(i, len(replies) - 1)])
        step = convo.advance()
        i += 1
    assert step["done"] is True
    assert convo.is_satisfied() is True
    draft = step["draft"]
    assert draft["sentiment"] == "positive"
    assert draft["suggested_rating"] == 5
    assert draft["text"]


def test_conversation_credits_unprompted_theme():
    convo = _convo()
    convo.advance()  # opening (pending = OVERALL)
    # Customer volunteers the breakout theme in their opening reply.
    convo.ingest_user_message("Love it, and it gave me zero acne at all!")
    assert "breakouts/acne" in convo.covered
    # Next question should therefore skip breakouts.
    step = convo.advance()
    assert convo.pending_issue != "breakouts/acne"


def test_conversation_handles_decline_gracefully():
    convo = _convo()
    convo.advance()
    convo.ingest_user_message("No thanks, not interested.")
    step = convo.advance()
    assert step["done"] is True
    assert convo.is_satisfied() is True
    assert convo.review_draft()["declined"] is True


def test_conversation_negative_sentiment_maps_to_low_rating():
    convo = _convo()
    step = convo.advance()
    i = 0
    while not step["done"] and i < 10:
        convo.ingest_user_message("It broke me out horribly, I hate it and returned it.")
        step = convo.advance()
        i += 1
    draft = convo.review_draft()
    assert draft["sentiment"] == "negative"
    assert draft["suggested_rating"] == 2
