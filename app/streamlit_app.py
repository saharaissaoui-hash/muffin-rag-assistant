"""
streamlit_app.py

Streamlit UI for the Chef Muffin RAG assistant.

Features:
- chat UX with session history
- sidebar controls (top_k, reveal sources)
- quick prompt chips
- two-tab output (Answer / Retrieved recipes)
- graceful error message if Ollama is not running
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.generation.chef_muffin import answer  # noqa: E402
from src.retrieval.retrieve import retrieve  # noqa: E402


st.set_page_config(page_title="Chef Muffin", page_icon="🧁", layout="centered")


CSS = """
<style>
.block-container { padding-top: 1.8rem; max-width: 980px; }

.hero {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px 18px;
  background: radial-gradient(1200px 200px at 20% 0%, rgba(255,255,255,0.06), transparent),
              rgba(255,255,255,0.03);
  margin-bottom: 16px;
}

.hero h1 {
  margin: 0;
  letter-spacing: -0.03em;
  font-size: 2.0rem;
}

.hero p {
  margin: 6px 0 0 0;
  color: rgba(255,255,255,0.70);
  font-size: 1.0rem;
  line-height: 1.35rem;
}

.kpi-row { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.kpi {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px;
  padding: 6px 12px;
  background: rgba(255,255,255,0.02);
  font-size: 0.9rem;
  color: rgba(255,255,255,0.75);
}

.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.03);
  margin: 10px 0;
}

.card-title { font-weight: 650; margin-bottom: 6px; }
.card-sub { color: rgba(255,255,255,0.70); font-size: 0.95rem; line-height: 1.25rem; }

.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 0.82rem;
  margin-right: 8px;
  color: rgba(255,255,255,0.72);
}

hr { border: none; height: 1px; background: rgba(255,255,255,0.12); margin: 14px 0; }

div[data-testid="stChatMessage"] { padding-top: 4px; padding-bottom: 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_query", "")


def ollama_healthcheck() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=1.0)
        return r.status_code < 500
    except Exception:
        return False


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Chef Muffin</h1>
          <p>Assistant culinaire spécialisé exclusivement dans les muffins. Décris ta demande ou tes ingrédients,
          je récupère les meilleures recettes de ton corpus et je génère une recommandation en français.</p>
          <div class="kpi-row">
            <span class="kpi">Mode: Muffins only</span>
            <span class="kpi">RAG: ChromaDB</span>
            <span class="kpi">Embeddings: multilingual MiniLM</span>
            <span class="kpi">LLM: Ollama (mistral)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls():
    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Recipes retrieved", 1, 8, 3)
        show_sources = st.toggle("Show retrieved recipes", value=True)
        st.markdown("---")
        st.caption("Tip: keep Ollama running (`ollama serve`).")
        if st.button("Clear chat"):
            st.session_state.history = []
            st.session_state.last_sources = []
            st.session_state.last_query = ""
            st.rerun()
    return top_k, show_sources


def quick_chips() -> str | None:
    st.markdown("Try a quick prompt:")
    cols = st.columns(3)
    prompts = [
        "Je veux un muffin au chocolat.",
        "J'ai des bananes et du yaourt, propose-moi un muffin.",
        "J'ai du chèvre et des épinards, je veux un muffin salé.",
        "Je veux un muffin sans trop de sucre.",
        "J'ai des myrtilles, propose un muffin moelleux.",
        "Je veux un muffin très simple, ingrédients basiques.",
    ]
    clicked = None
    for i, p in enumerate(prompts):
        with cols[i % 3]:
            if st.button(p, use_container_width=True):
                clicked = p
    st.markdown("<hr/>", unsafe_allow_html=True)
    return clicked


def render_chat() -> None:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def render_sources(sources):
    if not sources:
        st.info("No recipes retrieved yet.")
        return

    for idx, s in enumerate(sources, start=1):
        title = s.get("title", "")
        ingredients = s.get("ingredients", "")
        distance = s.get("distance", None)

        meta_line = f"<span class='badge'>#{idx}</span>"
        if isinstance(distance, (int, float)):
            meta_line += f"<span class='badge'>distance: {distance:.4f}</span>"

        st.markdown(
            f"""
            <div class="card">
              <div>{meta_line}</div>
              <div class="card-title">{title}</div>
              <div class="card-sub">{ingredients}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    init_state()
    hero()
    top_k, show_sources = sidebar_controls()

    if not ollama_healthcheck():
        st.error(
            "Ollama is not reachable on localhost:11434. "
            "Start it with `ollama serve`, then refresh this page."
        )
        st.stop()

    clicked_prompt = quick_chips()

    render_chat()

    user_text = st.chat_input("Example: J'ai du chocolat noir, des oeufs et du lait. Je veux un muffin.")
    if clicked_prompt and not user_text:
        user_text = clicked_prompt

    if not user_text:
        return

    user_text = user_text.strip()
    st.session_state.last_query = user_text
    st.session_state.history.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        tab_answer, tab_sources = st.tabs(["Answer", "Retrieved recipes"])

        with tab_answer:
            with st.spinner("Retrieving recipes..."):
                sources = retrieve(user_text, top_k=top_k)
                st.session_state.last_sources = sources

            with st.spinner("Generating response..."):
                response = answer(user_text, top_k=top_k)

            st.write(response)

        with tab_sources:
            if show_sources:
                render_sources(st.session_state.last_sources)
            else:
                st.info("Enable 'Show retrieved recipes' in the sidebar to display sources.")

    st.session_state.history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
