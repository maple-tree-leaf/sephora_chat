"""
app.py — Streamlit web UI for the Sephora reviews stats + RAG chatbot.

SETUP (one time):
    pip install -r requirements.txt

    Get a free API key at https://console.groq.com/keys, then set it as an
    environment variable before running locally:
        Windows (PowerShell):  $env:GROQ_API_KEY="your-key-here"
        Mac/Linux:              export GROQ_API_KEY="your-key-here"

    If deploying on Streamlit Community Cloud, add GROQ_API_KEY under
    "Secrets" in the app's settings instead — never commit the key to GitHub.

RUN:
    streamlit run app.py

Then open the browser tab it pops up (usually http://localhost:8501).
Use the sidebar to point at your data folder and build the pipeline once;
after that it's cached and you just chat.
"""

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import review_agent
import sephora_core as core

# If GROQ_API_KEY isn't set as an env var (typical when running locally),
# fall back to Streamlit secrets (typical when deployed on Community Cloud)
# and export it so the rest of the app / core module can just read os.environ.
if not os.environ.get("GROQ_API_KEY"):
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
        if secret_key:
            os.environ["GROQ_API_KEY"] = secret_key
    except Exception:
        pass

EXAMPLE_PROMPTS = [
    "What are the most common complaints overall?",
    "What's most common in June for Moisturizer?",
    "Why do people say this product didn't work?",
]

st.set_page_config(page_title="Sephora Reviews Q&A", page_icon="🧴", layout="wide")
st.title("🧴 Sephora Skincare Product Reviews — Stats + RAG Chatbot")
st.caption("Stat-style questions are answered from precomputed counts. Open-ended questions use semantic search over a cached review sample.")

# ----------------------------------------------------------------------
# SIDEBAR: setup
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    data_dir = st.text_input("Data folder (with product_info.csv + reviews*.csv)", value="./data")
    llm_model = st.selectbox(
        "Groq model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
        index=0,
    )
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("No GROQ_API_KEY found. Set it as an environment variable "
                   "(local) or a Streamlit secret (deployed) before chatting.")
    use_llm_tagger = st.checkbox(
        "Use LLM-based issue tagging (slower, more accurate)",
        value=False,
        help="If unchecked, issues are tagged with fast keyword matching. "
             "If checked, the Groq API reads each negative review and assigns issue "
             "categories — better recall on phrasing the keyword list misses, "
             "but takes longer on first build and uses more API calls.",
    )
    rebuild = st.button("Build / Rebuild pipeline", type="primary")
    st.divider()
    st.caption(f"Cached sample: {'found' if os.path.exists(core.SAMPLE_PATH) else 'not found'}")
    st.caption(f"Cached vector index: {'found' if os.path.exists(core.CHROMA_DIR) else 'not found'}")


# ----------------------------------------------------------------------
# PIPELINE BUILD / LOAD (cached in session so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline(data_dir, use_llm_tagger, llm_model, force_rebuild, _progress_cb=None):
    have_raw_data = os.path.exists(os.path.join(data_dir, "product_info.csv"))
    used_raw_data = False

    if force_rebuild and have_raw_data:
        # Full rebuild from raw CSVs — only possible when the raw dataset
        # actually exists on disk (e.g. running locally with the real data/
        # folder). On a cloud host this folder won't exist on purpose, since
        # the full ~1M-row dataset is too large to commit to GitHub.
        raw, n_reviews, n_products = core.load_raw(data_dir)
        sample = core.build_sample(raw)
        sample.to_parquet(core.SAMPLE_PATH)
        used_raw_data = True
    elif os.path.exists(core.SAMPLE_PATH):
        # No raw data available (or rebuild not requested) — use the
        # pre-built sample that was committed to the repo. This is the path
        # a cloud deployment should always take.
        sample = pd.read_parquet(core.SAMPLE_PATH)
    else:
        raise FileNotFoundError(
            f"No raw data found at '{data_dir}' and no cached '{core.SAMPLE_PATH}' "
            f"in the repo either. If running locally, point data_dir at your real "
            f"dataset folder. If deployed, make sure {core.SAMPLE_PATH} was committed "
            f"to GitHub alongside app.py."
        )

    # A rebuild should always re-tag issues and reindex with the current
    # settings, even when we fell back to the cached sample above — otherwise
    # toggling "Use LLM-based issue tagging" and clicking Rebuild would appear
    # to do nothing.
    aggs = core.build_aggregates(sample, use_llm_tagger=use_llm_tagger, llm_model=llm_model,
                                  progress_cb=_progress_cb, force_retag=force_rebuild)
    categories = sample["category"].dropna().unique().tolist()

    if force_rebuild or not os.path.exists(core.CHROMA_DIR):
        model, collection = core.build_index(sample)
    else:
        try:
            model, collection = core.load_index()
        except Exception:
            model, collection = core.build_index(sample)

    return sample, aggs, categories, model, collection, have_raw_data, used_raw_data


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if rebuild:
    progress_placeholder = st.empty()

    def _progress_cb(done, total):
        progress_placeholder.progress(done / total if total else 0,
                                       text=f"Tagging negative reviews... {done}/{total}")

    with st.spinner("Loading data, sampling, tagging issues, and building vector index... this can take a few minutes the first time."):
        get_pipeline.clear()  # force a fresh build
        st.session_state.pipeline = get_pipeline(data_dir, use_llm_tagger, llm_model, True,
                                                  _progress_cb=_progress_cb)
    progress_placeholder.empty()

    used_raw_data = st.session_state.pipeline[-1]
    if not used_raw_data:
        st.info(
            f"No raw dataset found at '{data_dir}', so the pipeline was rebuilt from the "
            f"cached sample ({core.SAMPLE_PATH}) instead — issue tags and the vector index "
            f"were refreshed with your current settings, but the underlying review sample "
            f"itself is unchanged."
        )
    st.success("Pipeline ready.")
elif st.session_state.pipeline is None and os.path.exists(core.SAMPLE_PATH):
    with st.spinner("Loading cached sample and building/loading vector index..."):
        st.session_state.pipeline = get_pipeline(data_dir, use_llm_tagger, llm_model, False)

# ----------------------------------------------------------------------
# MAIN: stats overview + chat
# ----------------------------------------------------------------------
if st.session_state.pipeline is None:
    st.info("👈 Set your data folder in the sidebar and click **Build / Rebuild pipeline** to get started.")
    st.stop()

sample, aggs, categories, model, collection, _, _ = st.session_state.pipeline

tab_chat, tab_stats, tab_outreach = st.tabs(
    ["💬 Chat", "📊 Stats overview", "📣 Review outreach"]
)

with tab_stats:
    st.subheader("Sample summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews in sample", f"{len(sample):,}")
    c2.metric("Categories", len(categories))
    c3.metric("Negative reviews (rating ≤2)", f"{(sample['rating'] <= 2).sum():,}")

    st.subheader("Most common issues overall")
    overall_df = pd.DataFrame(aggs["overall"].most_common(), columns=["issue", "count"])
    st.bar_chart(overall_df.set_index("issue"))

    st.subheader("Issues by category")
    category_filter = st.selectbox("Filter by category", ["All categories"] + sorted(categories))
    by_category = aggs["by_category"]
    if category_filter != "All categories":
        by_category = by_category[by_category["category"] == category_filter]
    if not by_category.empty:
        if category_filter == "All categories":
            pivot = by_category.pivot_table(index="issues", columns="category", values="count", fill_value=0)
        else:
            pivot = by_category.set_index("issues")["count"]
        st.bar_chart(pivot)
    st.dataframe(by_category, width='stretch')

    if not aggs["by_month"].empty:
        st.subheader("Issues by month")
        top_issues = [i for i, _ in aggs["overall"].most_common(6)]
        month_pivot = (aggs["by_month"][aggs["by_month"]["issues"].isin(top_issues)]
                       .pivot_table(index="month", columns="issues", values="count", fill_value=0))
        st.line_chart(month_pivot)
        st.dataframe(aggs["by_month"], width='stretch')

with tab_chat:
    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "busy" not in st.session_state:
        st.session_state.busy = False

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    def _submit_question(q):
        st.session_state.busy = True
        st.session_state.pending_question = q
        st.session_state.messages.append(("user", q))

    if not st.session_state.messages:
        st.caption("Try one of these, or ask your own question below:")
        cols = st.columns(len(EXAMPLE_PROMPTS))
        for col, prompt_text in zip(cols, EXAMPLE_PROMPTS):
            if col.button(prompt_text, disabled=st.session_state.busy, width='stretch'):
                _submit_question(prompt_text)
                st.rerun()

    # Render chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            if isinstance(content, tuple):
                text, sources = content
                st.write(text)
                if sources:
                    with st.expander(f"Sources ({len(sources)} reviews)"):
                        for s in sources:
                            st.markdown(
                                f"**{s.get('product_name') or 'Unknown product'}** "
                                f"— category: {s.get('category') or 'n/a'}, rating: {s.get('rating') or 'n/a'}"
                            )
                            st.caption(s.get("text", ""))
            else:
                st.write(content)

    # Disable input while a question is being processed
    question = st.chat_input(
        "Ask about product issues, e.g. 'What's most common in June for Moisturizer?'",
        disabled=st.session_state.busy,
    )

    # PASS 1: a new question just came in and nothing is in flight yet.
    # Lock the input and store the user's message, then immediately rerun
    # so the disabled state actually paints to the browser before any
    # work happens — without this rerun, "disabled" and "answer generated"
    # would both happen in the same script pass and never be visible.
    if question and not st.session_state.busy:
        _submit_question(question)
        st.rerun()

    # PASS 2: input is locked and there's a pending question waiting on an
    # answer — this only runs on the rerun triggered above, so the disabled
    # chat box has already been rendered to the browser by this point.
    if st.session_state.busy and st.session_state.pending_question:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = [
                        (role, content if isinstance(content, str) else content[0])
                        for role, content in st.session_state.messages[:-1]
                    ]
                    resp, sources = core.answer(
                        st.session_state.pending_question,
                        aggs,
                        categories,
                        model,
                        collection,
                        llm_model=llm_model,
                        history=history,
                    )
                except Exception as e:
                    resp, sources = f"Error: {e}", []
            st.write(resp)
            if sources:
                with st.expander(f"Sources ({len(sources)} reviews)"):
                    for s in sources:
                        st.markdown(
                            f"**{s.get('product_name') or 'Unknown product'}** "
                            f"— category: {s.get('category') or 'n/a'}, rating: {s.get('rating') or 'n/a'}"
                        )
                        st.caption(s.get("text", ""))

        st.session_state.messages.append(("assistant", (resp, sources)))
        st.session_state.pending_question = None
        st.session_state.busy = False
        st.rerun()

with tab_outreach:
    st.subheader("Proactive review-solicitation agent")
    st.caption(
        "Simulate reaching out to a customer ~2 weeks after purchase. The agent "
        "reads how past buyers reviewed the same product, turns the recurring "
        "themes into targeted questions, and adapts as the customer replies until "
        "it has a satisfactory, well-rounded review."
    )

    products = sorted(sample["product_name"].dropna().unique().tolist()) \
        if "product_name" in sample.columns else []

    setup_col, date_col = st.columns([2, 1])
    with setup_col:
        selected_product = st.selectbox(
            "Product the customer purchased",
            products or ["(no product names in sample)"],
            disabled=not products,
        )
    with date_col:
        default_purchase = date.today() - timedelta(days=review_agent.OUTREACH_MIN_DAYS)
        purchase_date = st.date_input("Purchase date", value=default_purchase)

    days = review_agent.days_since_purchase(purchase_date)
    due = review_agent.is_due_for_outreach(purchase_date)
    if due:
        st.success(f"✅ Purchased {days} days ago — due for outreach "
                   f"(threshold: {review_agent.OUTREACH_MIN_DAYS} days).")
    else:
        eta = review_agent.next_outreach_date(purchase_date)
        st.info(f"⏳ Purchased {days} days ago — too soon. The agent would wait "
                f"until {eta:%b %d, %Y} ({review_agent.OUTREACH_MIN_DAYS} days after purchase).")

    if products:
        insights = review_agent.product_review_insights(sample, product_name=selected_product)
        with st.expander("What the database knows about this product", expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("Past reviews used", insights["n_reviews"])
            m2.metric("Avg rating", insights["avg_rating"] if insights["avg_rating"] is not None else "n/a")
            m3.metric("Grounding scope", insights["scope"])
            if insights["scope"] == "category":
                st.caption("Too few product-specific reviews, so themes are drawn from the "
                           "product's category instead.")
            if insights["top_issues"]:
                st.write("**Anticipated themes to probe:**")
                for issue in insights["top_issues"][:review_agent.MAX_TALKING_POINTS]:
                    st.write(f"- {issue} ({insights['issue_counts'][issue]} mentions)")
            else:
                st.write("No recurring negative themes found — the agent will keep it open-ended.")

    # -- conversation state (kept separate from the analyst chat) -------
    start_disabled = not products or not due
    if st.button("Start outreach conversation", type="primary", disabled=start_disabled):
        convo = review_agent.ReviewConversation(
            review_agent.product_review_insights(sample, product_name=selected_product),
            llm_model=llm_model,
        )
        first = convo.advance()
        st.session_state.ro_convo = convo
        st.session_state.ro_log = [("assistant", first["message"])]
        st.session_state.ro_done = first["done"]
        st.session_state.ro_draft = first.get("draft")
        st.rerun()

    if start_disabled and not due:
        st.caption("Outreach is only offered once the 2-week threshold has passed. "
                   "Pick an earlier purchase date to try it.")

    convo = st.session_state.get("ro_convo")
    if convo is not None:
        for role, content in st.session_state.ro_log:
            with st.chat_message("assistant" if role == "assistant" else "user"):
                st.write(content)

        if not st.session_state.get("ro_done"):
            reply = st.chat_input("Reply as the customer...")
            if reply:
                convo.ingest_user_message(reply)
                st.session_state.ro_log.append(("user", reply))
                step = convo.advance()
                st.session_state.ro_log.append(("assistant", step["message"]))
                st.session_state.ro_done = step["done"]
                st.session_state.ro_draft = step.get("draft")
                st.rerun()
        else:
            draft = st.session_state.get("ro_draft")
            if draft and not draft.get("declined"):
                st.divider()
                st.subheader("📝 Collected review draft")
                d1, d2 = st.columns(2)
                d1.metric("Suggested rating", f"{draft['suggested_rating']} / 5"
                          if draft["suggested_rating"] else "n/a")
                d2.metric("Overall sentiment", draft["sentiment"] or "n/a")
                st.text_area("Review text", draft["text"], height=160)
            if st.button("Reset conversation"):
                for k in ("ro_convo", "ro_log", "ro_done", "ro_draft"):
                    st.session_state.pop(k, None)
                st.rerun()
