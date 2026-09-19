import io
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_pdf
from src.retrieval import KnowledgeBase
from src.llm import generate_answer

load_dotenv()

st.set_page_config(
    page_title="KnowFlow AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
:root { --border:#e2e8f0; --muted:#64748b; --panel:#f8fafc; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.hero { padding: 0.25rem 0 1rem 0; }
.hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -0.03em; }
.hero p { margin: 0.35rem 0 0 0; color: var(--muted); font-size: 1.02rem; }
.metric-card {
    padding: 0.85rem 1rem; border:1px solid var(--border);
    border-radius: 12px; background:white;
}
.source-card {
    border:1px solid var(--border); border-radius:10px;
    padding:0.8rem; margin:0.45rem 0; background:var(--panel);
}
.badge {
    display:inline-block; padding:0.18rem 0.5rem; border-radius:999px;
    background:#eef2ff; font-size:0.76rem; font-weight:600;
}
.small { color:var(--muted); font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session state
# ----------------------------
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": 5,
        "candidate_k": 12,
        "rerank": True,
        "abstain": 0.28,
    }

kb = st.session_state.kb
settings = st.session_state.settings

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🧠 KnowFlow AI Pro")
    st.caption("Production-style document intelligence using RAG")
    st.divider()

    uploaded = st.file_uploader(
        "Upload knowledge documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Use documents you are permitted to use for the assignment.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📥 Index", use_container_width=True, disabled=not uploaded):
            new_names = [f.name for f in uploaded]
            with st.status("Building knowledge base...", expanded=True) as status:
                try:
                    kb.clear()
                    total = 0
                    pages = 0
                    for f in uploaded:
                        data = f.read()
                        chunks, meta = ingest_pdf(
                            data,
                            source_name=f.name,
                            chunk_size=650,
                            overlap=110,
                        )
                        kb.add_chunks(chunks)
                        total += len(chunks)
                        pages += meta["pages"]
                        st.write(f"Indexed **{f.name}** → {len(chunks)} chunks")
                    kb.build()
                    st.session_state.docs = new_names
                    st.session_state.messages = []
                    st.session_state.last_results = []
                    status.update(
                        label=f"Ready — {len(new_names)} document(s), {total} chunks",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label="Indexing failed", state="error")
                    st.error(str(e))

    with col_b:
        if st.button("🗑️ Clear", use_container_width=True):
            kb.clear()
            st.session_state.docs = []
            st.session_state.messages = []
            st.session_state.last_results = []
            st.rerun()

    st.divider()
    st.markdown("### Retrieval controls")
    settings["top_k"] = st.slider("Final chunks", 3, 8, settings["top_k"])
    settings["candidate_k"] = st.slider("Candidate pool", 8, 20, settings["candidate_k"])
    settings["rerank"] = st.toggle(
        "Use cross-encoder reranking",
        value=settings["rerank"],
        help="Falls back to hybrid retrieval if the reranker model cannot be loaded."
    )
    settings["abstain"] = st.slider(
        "Grounding threshold",
        0.05, 0.75, settings["abstain"], 0.01,
        help="Below this retrieval score, the assistant is encouraged to abstain."
    )

    st.divider()
    st.markdown("### Knowledge base")
    st.write(f"**Documents:** {len(st.session_state.docs)}")
    st.write(f"**Indexed chunks:** {kb.count}")
    st.caption("Dense + lexical retrieval, metadata-aware citations, grounded generation.")

# ----------------------------
# Header and metrics
# ----------------------------
st.markdown(
    '<div class="hero"><h1>KnowFlow AI Pro</h1>'
    '<p>Evidence-first RAG assistant: retrieve the right context, answer from it, and show where the answer came from.</p></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><b>{len(st.session_state.docs)}</b><br><span class="small">Documents</span></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><b>{kb.count}</b><br><span class="small">Indexed chunks</span></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><b>{len(st.session_state.messages)}</b><br><span class="small">Messages</span></div>', unsafe_allow_html=True)
with m4:
    provider = os.getenv("LLM_PROVIDER", "gemini").upper()
    st.markdown(f'<div class="metric-card"><b>{provider}</b><br><span class="small">LLM provider</span></div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat", "🔎 Retrieval Inspector", "🧪 Evaluation", "📘 Explain"])

# ----------------------------
# Chat tab
# ----------------------------
with tabs[0]:
    if not st.session_state.docs:
        st.info("Upload your PDFs in the sidebar, click **Index**, then start asking questions.")
        st.markdown("""
**Recommended demo sequence**
1. Ask a direct document question.
2. Ask the same fact using different wording.
3. Ask a question requiring multiple documents.
4. Ask something absent from the documents and demonstrate abstention.
""")

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Evidence ({len(msg['sources'])})", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card"><b>{s["label"]}</b>'
                            f'<br><span class="small">{s["preview"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                c1, c2, c3 = st.columns([1, 1, 5])
                with c1:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state.feedback.append({
                            "time": datetime.now().isoformat(),
                            "message_index": idx,
                            "feedback": "up"
                        })
                with c2:
                    if st.button("👎", key=f"down_{idx}"):
                        st.session_state.feedback.append({
                            "time": datetime.now().isoformat(),
                            "message_index": idx,
                            "feedback": "down"
                        })

    if question := st.chat_input("Ask a question about your documents…"):
        if kb.count == 0:
            st.warning("Index at least one PDF first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating a grounded answer…"):
                results = kb.search(
                    question,
                    candidate_k=settings["candidate_k"],
                    final_k=settings["top_k"],
                    rerank=settings["rerank"],
                )
                st.session_state.last_results = results

                answer = generate_answer(
                    question=question,
                    retrieved=results,
                    history=st.session_state.messages[-8:],
                    grounding_threshold=settings["abstain"],
                )

            st.markdown(answer)

            sources = []
            for i, r in enumerate(results, 1):
                sources.append({
                    "label": f'[S{i}] {r["source"]} — page {r["page"]} • retrieval {r["score"]:.3f}',
                    "preview": r["text"][:420].replace("\n", " ") + ("…" if len(r["text"]) > 420 else ""),
                })

            if sources:
                with st.expander("📚 Evidence used", expanded=True):
                    for s in sources:
                        st.markdown(
                            f'<div class="source-card"><b>{s["label"]}</b>'
                            f'<br><span class="small">{s["preview"]}</span></div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

# ----------------------------
# Retrieval inspector
# ----------------------------
with tabs[1]:
    st.markdown("### 🔎 Retrieval Inspector")
    st.caption("Inspect what the RAG system retrieved before the LLM generated the answer.")

    query = st.text_input("Test retrieval query", key="inspector_query")
    if st.button("Run retrieval test", disabled=not query or kb.count == 0):
        results = kb.search(
            query,
            candidate_k=settings["candidate_k"],
            final_k=settings["top_k"],
            rerank=settings["rerank"],
        )
        st.session_state.last_results = results

    if st.session_state.last_results:
        for i, r in enumerate(st.session_state.last_results, 1):
            st.markdown(
                f"**{i}. {r['source']} — page {r['page']}** "
                f"`score={r['score']:.3f}` `dense={r.get('dense_score', 0):.3f}` `lexical={r.get('lexical_score', 0):.3f}`"
            )
            st.progress(min(max((r["score"] + 1) / 2, 0), 1))
            st.write(r["text"])
            st.divider()

# ----------------------------
# Evaluation
# ----------------------------
with tabs[2]:
    st.markdown("### 🧪 Lightweight evaluation")
    st.caption("Run retrieval tests against a small question set. Do not report metrics you have not actually measured.")

    eval_path = Path("evaluation/test_questions.json")
    if eval_path.exists():
        tests = json.loads(eval_path.read_text(encoding="utf-8"))
        st.write(f"Included test cases: **{len(tests)}**")
        for i, t in enumerate(tests, 1):
            st.markdown(f"**{i}.** {t['question']}")
            st.caption(f"Expected behavior: {t.get('expected_behavior', 'Not specified')}")

        st.info("For the actual submission, add questions whose expected source/page you know. Then record Recall@K or hit-rate from real runs.")
    else:
        st.info("Add evaluation/test_questions.json to define your test set.")

    if st.session_state.feedback:
        st.markdown("#### User feedback captured this session")
        ups = sum(x["feedback"] == "up" for x in st.session_state.feedback)
        downs = sum(x["feedback"] == "down" for x in st.session_state.feedback)
        st.write(f"👍 {ups}   |   👎 {downs}")

# ----------------------------
# Explain tab
# ----------------------------
with tabs[3]:
    st.markdown("### 📘 Explain the system")
    st.markdown("""
**1. Ingestion**  
PyMuPDF extracts page-level text from PDFs. Each chunk keeps the document name and page number.

**2. Chunking**  
Long pages are split into overlapping chunks so retrieval can return focused context without losing boundary information.

**3. Embeddings**  
A Sentence-Transformer converts each chunk and the user query into vectors.

**4. Retrieval**  
KnowFlow combines semantic vector retrieval with a lexical signal. A configurable cross-encoder can rerank the candidate pool.

**5. Grounded generation**  
The LLM receives the question plus retrieved evidence. The prompt explicitly treats document text as untrusted reference material, not as instructions, reducing prompt-injection risk.

**6. Citations**  
Each retrieved chunk carries its source filename and page, so the interface can show evidence alongside the answer.

**7. Abstention**  
When retrieval is weak, the assistant is instructed to say that the information was not found rather than confidently inventing an answer.
""")

st.download_button(
    "⬇️ Download conversation JSON",
    data=json.dumps(st.session_state.messages, indent=2),
    file_name="knowflow_conversation.json",
    mime="application/json",
    disabled=not st.session_state.messages,
)

st.download_button(
    "⬇️ Download feedback JSON",
    data=json.dumps(st.session_state.feedback, indent=2),
    file_name="knowflow_feedback.json",
    mime="application/json",
    disabled=not st.session_state.feedback,
)
