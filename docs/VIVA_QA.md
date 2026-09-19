# Round 2 viva preparation

## What is RAG?
Retrieval-Augmented Generation combines retrieval of relevant external context with LLM generation. The model answers using the retrieved evidence instead of relying only on its internal knowledge.

## Why not fine-tuning?
The task is document-grounded question answering. RAG is a natural fit because the source documents can change without retraining the language model.

## What are embeddings?
Embeddings represent text as vectors in a numerical space where semantically similar texts tend to be closer.

## Why FAISS?
FAISS provides efficient similarity search over embedding vectors and is simple to integrate for a local prototype.

## Why chunk documents?
Whole documents are often too large and too noisy to retrieve precisely. Chunks let the retriever bring focused evidence into the LLM context.

## Why overlap?
Overlap reduces the chance that a concept split across two neighboring chunks loses essential context.

## Why hybrid retrieval?
Dense retrieval handles semantic paraphrases well. Lexical retrieval helps with exact terms, names, IDs, and phrases. Combining them can make retrieval more robust.

## Why reranking?
The initial retriever creates a candidate set. A cross-encoder can inspect the query and each candidate together and reorder the candidates by relevance.

## How do you reduce hallucination?
Grounded prompt, explicit abstention behavior, evidence citations, retrieval inspection, and testing with unanswerable questions.

## What if the document contains malicious instructions?
Retrieved document text is presented as untrusted reference material, and the prompt instructs the model never to follow instructions inside those references.

## Biggest limitation?
A scanned PDF may have little or no extractable text. An OCR pipeline would be needed for robust scanned-document support.

## What did you learn?
A strong RAG system is not just “PDF + LLM.” Retrieval quality, chunking, metadata, grounding, failure handling, and evaluation matter as much as the final generation step.
