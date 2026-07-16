"""
Unit tests for the pure logic in sephora_core.py: keyword tagging, routing,
month/category extraction, sample building, and stat context formatting.

These deliberately avoid the Groq API, sentence-transformers, and chromadb so
they run fast and without network access or a GROQ_API_KEY.

Run with:
    pytest tests/
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sephora_core as core


# ----------------------------------------------------------------------
# keyword_tag_issues
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected_issue", [
    ("This broke me out badly, so many pimples now", "breakouts/acne"),
    ("My skin felt like it was on fire, so much stinging", "irritation/redness"),
    ("Left my face super dry and flaking", "dryness/flaking"),
    ("Way too oily and greasy on my skin", "oily/greasy"),
    ("Smells absolutely terrible, like chemicals", "bad smell"),
    ("The pump broke and it leaked all over my bag", "packaging issues"),
    ("Way overpriced for what you get", "too expensive"),
    ("Did not work at all, total waste of money", "no effect / didn't work"),
    ("Got hives and a bad allergic reaction", "allergic reaction"),
    ("Super sticky and gritty texture", "texture/consistency"),
])
def test_keyword_tag_issues_detects_expected_category(text, expected_issue):
    assert expected_issue in core.keyword_tag_issues(text)


def test_keyword_tag_issues_avoids_substring_false_positives():
    # "capacity" contains "cap" but shouldn't be tagged as a packaging issue,
    # and "costume" contains "cost" but shouldn't be tagged as "too expensive".
    assert "packaging issues" not in core.keyword_tag_issues("Great bottle capacity, holds a lot")
    assert "too expensive" not in core.keyword_tag_issues("Bought this for a Halloween costume")


def test_keyword_tag_issues_stems_still_match_inflections():
    assert "dryness/flaking" in core.keyword_tag_issues("This is so drying on my skin")
    assert "irritation/redness" in core.keyword_tag_issues("Caused a lot of irritation and redness")


def test_keyword_tag_issues_no_match_returns_empty_list():
    assert core.keyword_tag_issues("I love this product, works great!") == []


# ----------------------------------------------------------------------
# extract_month / extract_category
# ----------------------------------------------------------------------
def test_extract_month_finds_month_name_case_insensitive():
    assert core.extract_month("What happened in June for this product?") == 6
    assert core.extract_month("Complaints in DECEMBER") == 12


def test_extract_month_returns_none_when_absent():
    assert core.extract_month("What are the most common issues?") is None


def test_extract_category_matches_known_category():
    categories = ["Moisturizer", "Cleanser", "Serum"]
    assert core.extract_category("Issues with Moisturizer in June", categories) == "Moisturizer"


def test_extract_category_returns_none_when_absent():
    categories = ["Moisturizer", "Cleanser"]
    assert core.extract_category("What do people think overall?", categories) is None


# ----------------------------------------------------------------------
# is_stat_question routing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "What are the most common issues?",
    "How many reviews mention breakouts?",
    "Give me a breakdown of complaints by month",
    "Compare Moisturizer and Cleanser issues",
])
def test_is_stat_question_true_for_stat_phrases(question):
    assert core.is_stat_question(question) is True


def test_is_stat_question_false_for_open_ended_question():
    assert core.is_stat_question("Why do people love this serum so much?") is False


def test_is_stat_question_true_when_category_plus_issue_word_present():
    categories = ["Moisturizer", "Cleanser"]
    assert core.is_stat_question("What problems do people have with Moisturizer?", categories) is True


def test_is_stat_question_does_not_false_positive_on_trend_substring():
    # "trendy" contains "trend" but isn't asking for a trend/stat.
    assert core.is_stat_question("This brand is really trendy right now") is False


# ----------------------------------------------------------------------
# build_sample
# ----------------------------------------------------------------------
def _make_raw_df(n=60):
    rows = []
    for i in range(n):
        rows.append({
            "review_text": f"review {i} " + ("bad breakout" if i % 3 == 0 else "loved it"),
            "rating": 1 if i % 3 == 0 else 5,
            "secondary_category": "Moisturizer" if i % 2 == 0 else "Cleanser",
            "review_date": "2024-01-15" if i % 2 == 0 else "2024-06-15",
            "product_id": f"p{i % 5}",
            "product_name": f"Product {i % 5}",
            "brand_name": "TestBrand",
        })
    return pd.DataFrame(rows)


def test_build_sample_renames_and_buckets_correctly():
    raw = _make_raw_df()
    sample = core.build_sample(raw, sample_size=1000, per_group_cap=50)
    assert "category" in sample.columns
    assert "review_text" in sample.columns
    assert "rating" in sample.columns
    assert set(sample["category"].unique()) <= {"Moisturizer", "Cleanser"}
    assert sample["month"].dropna().isin([1, 6]).all()


def test_build_sample_raises_without_text_column():
    df = pd.DataFrame({"rating": [1, 2, 3]})
    with pytest.raises(KeyError):
        core.build_sample(df)


# ----------------------------------------------------------------------
# build_aggregates (keyword tagger only — no network/API calls)
# ----------------------------------------------------------------------
def _make_tagged_sample():
    return pd.DataFrame({
        "review_text": [
            "Broke me out badly, so many pimples",
            "Way too expensive for such a small bottle",
            "Loved this, no issues at all",
            "Leaked all over my bag, packaging broke",
        ],
        "rating": [1, 2, 5, 1],
        "category": ["Moisturizer", "Moisturizer", "Cleanser", "Cleanser"],
        "month": [6, 6, 7, 7],
    })


def test_build_aggregates_counts_issues_from_negative_reviews_only():
    sample = _make_tagged_sample()
    aggs = core.build_aggregates(sample, use_llm_tagger=False)
    assert aggs["overall"]["breakouts/acne"] == 1
    assert aggs["overall"]["too expensive"] == 1
    assert aggs["overall"]["packaging issues"] == 1
    # the 5-star review is excluded even though tagging is skipped for it
    assert sum(aggs["overall"].values()) == 3


def test_build_aggregates_by_category_breakdown():
    sample = _make_tagged_sample()
    aggs = core.build_aggregates(sample, use_llm_tagger=False)
    by_cat = aggs["by_category"]
    moisturizer_issues = set(by_cat[by_cat["category"] == "Moisturizer"]["issues"])
    assert "breakouts/acne" in moisturizer_issues
    assert "too expensive" in moisturizer_issues


# ----------------------------------------------------------------------
# build_stat_context
# ----------------------------------------------------------------------
def test_build_stat_context_includes_category_when_mentioned():
    sample = _make_tagged_sample()
    aggs = core.build_aggregates(sample, use_llm_tagger=False)
    categories = ["Moisturizer", "Cleanser"]
    context = core.build_stat_context("What issues does Moisturizer have?", aggs, categories)
    assert "Moisturizer" in context


def test_build_stat_context_falls_back_to_overall_summary():
    sample = _make_tagged_sample()
    aggs = core.build_aggregates(sample, use_llm_tagger=False)
    context = core.build_stat_context("What are the most common issues overall?", aggs, [])
    assert "Overall most common issues" in context
