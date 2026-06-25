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

import pandas as pd
import streamlit as st

import sephora_core as core

st.set_page_config(page_title="Sephora Reviews Q&A", page_icon="🧴", layout="wide")
st.title("🧴 Sephora Skincare Product Reviews — Stats + RAG Chatbot")
st.caption("Stat-style questions are answered from precomputed counts. Open-ended questions use semantic search over a 10k-review sample.")

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
def get_pipeline(data_dir, use_llm_tagger, llm_model, force_rebuild):
    have_raw_data = os.path.exists(os.path.join(data_dir, "product_info.csv"))

    if force_rebuild and have_raw_data:
        # Full rebuild from raw CSVs — only possible when the raw dataset
        # actually exists on disk (e.g. running locally with the real data/
        # folder). On a cloud host this folder won't exist on purpose, since
        # the full ~1M-row dataset is too large to commit to GitHub.
        raw, n_reviews, n_products = core.load_raw(data_dir)
        sample = core.build_sample(raw)
        sample.to_parquet(core.SAMPLE_PATH)
    elif os.path.exists(core.SAMPLE_PATH):
        # No raw data available (or rebuild not requested) — use the
        # pre-built sample that was committed to the repo. This is the path
        # a cloud deployment should always take.
        sample = pd.read_parquet(core.SAMPLE_PATH)
    else:
        raise FileNotFoundError(
            f"No raw data found at '{data_dir}' and no cached '{core.SAMPLE_PATH}' "
            f"in the repo either. If running locally, point data_dir at your real "
            f"dataset folder. If deployed, make sure sample_10k.parquet was committed "
            f"to GitHub alongside app.py."
        )

    aggs = core.build_aggregates(sample, use_llm_tagger=use_llm_tagger, llm_model=llm_model)
    categories = sample["primary_category"].dropna().unique().tolist()

    if (force_rebuild and have_raw_data) or not os.path.exists(core.CHROMA_DIR):
        model, collection = core.build_index(sample)
    else:
        try:
            model, collection = core.load_index()
        except Exception:
            model, collection = core.build_index(sample)

    return sample, aggs, categories, model, collection


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if rebuild:
    with st.spinner("Loading data, sampling, tagging issues, and building vector index... this can take a few minutes the first time."):
        get_pipeline.clear()  # force a fresh build
        st.session_state.pipeline = get_pipeline(data_dir, use_llm_tagger, llm_model, True)
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

sample, aggs, categories, model, collection = st.session_state.pipeline

tab_chat, tab_stats = st.tabs(["💬 Chat", "📊 Stats overview"])

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
    st.dataframe(aggs["by_category"], use_container_width=True)

    if not aggs["by_month"].empty:
        st.subheader("Issues by month")
        st.dataframe(aggs["by_month"], use_container_width=True)

with tab_chat:
    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "busy" not in st.session_state:
        st.session_state.busy = False

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # Render chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
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
        st.session_state.busy = True
        st.session_state.pending_question = question
        st.session_state.messages.append(("user", question))
        st.rerun()

    # PASS 2: input is locked and there's a pending question waiting on an
    # answer — this only runs on the rerun triggered above, so the disabled
    # chat box has already been rendered to the browser by this point.
    if st.session_state.busy and st.session_state.pending_question:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = core.answer(
                        st.session_state.pending_question,
                        aggs,
                        categories,
                        model,
                        collection,
                        llm_model=llm_model,
                    )
                except Exception as e:
                    resp = f"Error: {e}"
            st.write(resp)

        st.session_state.messages.append(("assistant", resp))
        st.session_state.pending_question = None
        st.session_state.busy = False
        st.rerun()
