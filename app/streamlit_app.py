"""
streamlit_app.py

Streamlit UI for the Chef Muffin RAG assistant.

This app:
- collects a user query in French
- retrieves top-k muffin recipes from the vector store
- generates an answer using the local Ollama model
- optionally displays retrieved recipes for transparency/debug
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.generation.chef_muffin import answer  # noqa: E402
from src.retrieval.retrieve import retrieve  # noqa: E402


st.set_page_config(
    page_title="Chef Muffin",
    page_icon="🧁",
    layout="centered",
)


CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 900px; }
h1 { letter-spacing: -0.02em; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
  margin: 10px 0;
}
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 0.85rem;
  margin-right: 8px;
}
.hr {
  height: 1px;
  background: rgba(255,255,255,0.12);
  margin: 16px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {"role": "...", "content": "..."}
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = []


def render_header() -> None:
    st.title("Chef Muffin")
    st.markdown(
        '<div class="small-muted">Assistant culinaire spécialisé exclusivement dans les muffins. '
        "Pose une question ou liste tes ingrédients, et je te propose un muffin basé sur ton corpus.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def sidebar_controls():
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Recipes retrieved (top_k)", 1, 8, 3)
        show_sources = st.toggle("Show retrieved recipes", value=False)
        st.markdown("---")
        st.caption("Local LLM: Ollama (mistral).")
        if st.button("Clear chat history"):
            st.session_state.history = []
            st.session_state.last_retrieval = []
            st.rerun()
    return top_k, show_sources


def render_chat() -> None:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def render_sources(sources):
    if not sources:
        return

    st.subheader("Retrieved recipes")
    for idx, s in enumerate(sources, start=1):
        title = s.get("title", "")
        ingredients = s.get("ingredients", "")
        distance = s.get("distance", None)

        st.markdown(
            f"""
            <div class="card">
              <div>
                <span class="badge">#{idx}</span>
                <span class="badge">distance: {distance:.4f}</span>
              </div>
              <div style="margin-top:8px;"><b>{title}</b></div>
              <div style="margin-top:6px;" class="small-muted">{ingredients}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    init_state()
    render_header()

    top_k, show_sources = sidebar_controls()

    render_chat()

    user_text = st.chat_input("Ex: J'ai du chocolat noir, des oeufs et du lait. Je veux un muffin.")
    if not user_text:
        return

    st.session_state.history.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching muffin recipes..."):
            sources = retrieve(user_text, top_k=top_k)
            st.session_state.last_retrieval = sources

        with st.spinner("Generating answer..."):
            response = answer(user_text, top_k=top_k)

        st.write(response)

    st.session_state.history.append({"role": "assistant", "content": response})

    if show_sources:
        render_sources(st.session_state.last_retrieval)


if __name__ == "__main__":
    main()
