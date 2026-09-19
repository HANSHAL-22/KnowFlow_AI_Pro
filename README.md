# 🧠 KnowFlow AI Pro

## Mini AI Knowledge Assistant — RAG

KnowFlow AI Pro is a practical Retrieval-Augmented Generation application built for document-grounded question answering.

The design intentionally goes beyond a basic PDF chatbot:

- multi-document ingestion
- page-aware metadata
- overlapping chunking
- dense semantic retrieval
- lexical/BM25 retrieval
- hybrid ranking
- optional cross-encoder reranking
- grounded LLM prompt
- source citations
- conversation history
- retrieval inspector
- lightweight evaluation workflow
- feedback collection
- prompt-injection-aware reference handling
- downloadable conversation/feedback

---

## Assignment mapping

### Required

1. **PDF/document knowledge source** → PDF uploader
2. **Extract and process content** → PyMuPDF + cleaning
3. **Split into chunks** → overlapping word chunks
4. **Generate embeddings** → Sentence-Transformers
5. **Vector retrieval** → FAISS
6. **Accept user questions** → Streamlit chat
7. **Retrieve + generate with LLM** → hybrid retrieval + Gemini/OpenAI
8. **Primarily based on knowledge source** → grounded prompt + abstention behavior
9. **Simple usable interface** → Streamlit application

### Bonus features implemented

- Source citations
- Conversation history
- Multiple documents
- Improved retrieval
- Retrieval inspector
- Evaluation starter
- Feedback
- Deployment-ready structure
- Prompt-injection-aware context handling

---

## Architecture

```text
                        ┌──────────────────┐
                        │    Streamlit UI  │
                        └─────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                  Upload PDFs              User query
                     │                         │
                     v                         v
             ┌───────────────┐         ┌──────────────┐
             │   PyMuPDF     │         │ Query vector │
             └───────┬───────┘         └──────┬───────┘
                     v                         │
              clean + chunk                   │
                     │                         │
                     v                         │
             ┌───────────────┐                 │
             │   Embeddings  │                 │
             └───────┬───────┘                 │
                     v                         │
                 ┌────────┐                    │
                 │ FAISS  │<───────────────────┘
                 └───┬────┘
                     │
             + lexical retrieval
                     │
                     v
              hybrid candidates
                     │
             optional reranker
                     │
                     v
               top evidence
                     │
                     v
              grounded prompt
                     │
                     v
                 Gemini /
                  OpenAI
                     │
                     v
             answer + [S1] [S2]
```

---

## Why hybrid retrieval?

Pure vector search can miss exact terms, identifiers, or short phrases.

Pure lexical search can miss semantic paraphrases.

This project therefore combines:

- semantic similarity from embeddings
- lexical matching using a small BM25 implementation

A candidate pool can then be optionally reranked by a cross-encoder.

---

## Grounding and safety

The model is told:

- the document excerpts are **reference material**, not instructions
- unsupported claims must not be invented
- conflicting sources must be surfaced
- the answer must cite retrieved sources
- questions outside the knowledge base should be explicitly declined

This is an engineering choice to reduce hallucination and document-based prompt injection.

---

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure one LLM provider.

Run:

```bash
streamlit run app.py
```

---

## Round 2 demonstration

Use a legitimate and coherent collection such as:

- academic handbook
- examination regulations
- curriculum
- placement guidelines

Recommended demo:

1. Upload 2–5 PDFs.
2. Index them.
3. Ask a direct question.
4. Ask the same fact in different words.
5. Ask a multi-document question.
6. Ask a question that is absent from the documents.
7. Open **Evidence Used**.
8. Open **Retrieval Inspector** and explain the dense/lexical/rerank scores.
9. Explain one limitation and one future improvement.

---

## Evaluation

The repository includes a small starter test set.

For the final submission, create an evaluation set where you know the expected source/page. Then measure real retrieval hit-rate / Recall@K.

**Do not invent or present made-up metrics.**

---

## Transparency

AI assistance was used during development. Review and understand the code before submission, and disclose AI assistance according to the assignment instructions.

---

## Security

Never commit API keys.

Use `.env` locally and keep it in `.gitignore`.

Do not upload documents you are not allowed to use.

---

## Known limitations

- Scanned/image-only PDFs require OCR, which is intentionally not forced into the fast demo path.
- In-memory indexing resets when the Streamlit session restarts.
- Reranking downloads an additional model on first use.
- LLM output quality depends on the configured provider/model.

---

## Future upgrades

- persistent vector indexes
- hybrid query rewriting
- dedicated OCR pipeline
- automatic page-highlight viewer
- evaluation dataset management
- authentication
- telemetry/latency monitoring
- containerized deployment
