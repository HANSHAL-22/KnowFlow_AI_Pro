# 2–3 minute Round 2 demo

## Opening — 15 seconds

“KnowFlow AI Pro is a document-grounded RAG assistant. Instead of training a model on a private document, I retrieve the relevant evidence at query time and use that evidence to generate the answer.”

## Demo — about 90 seconds

1. Upload 2–5 legitimate documents.
2. Click Index.
3. Ask a direct question.
4. Show the answer with [S1]/[S2] evidence.
5. Ask the same question using different wording.
6. Ask something not covered by the documents.
7. Show the system declining to invent an answer.
8. Open Retrieval Inspector and explain dense + lexical retrieval and optional reranking.

## Close — 30 seconds

“My main design focus was not just getting an answer, but making the answer auditable. Every chunk carries document and page metadata, and the prompt explicitly treats retrieved text as untrusted reference material. If I had more time, I would add persistent indexes, OCR, a larger evaluation suite, and deeper monitoring.”
