import math
import re
from collections import Counter, defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25:
    """Small dependency-free BM25 implementation for hybrid retrieval."""
    def __init__(self, documents):
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1
        self.df = defaultdict(int)
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        self.n = len(self.docs)

    def score(self, query):
        q = tokenize(query)
        scores = [0.0] * self.n
        k1, b = 1.5, 0.75
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            dl = max(1, self.doc_len[i])
            for term in q:
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * dl / self.avgdl)
                scores[i] += idf * (freq * (k1 + 1)) / denom
        return np.asarray(scores, dtype="float32")


class KnowledgeBase:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.reranker = None
        self.chunks = []
        self.index = None
        self.bm25 = None

    @property
    def count(self):
        return len(self.chunks)

    def clear(self):
        self.chunks = []
        self.index = None
        self.bm25 = None

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def build(self):
        if not self.chunks:
            self.index = None
            self.bm25 = None
            return
        texts = [c["text"] for c in self.chunks]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25(texts)

    def _dense(self, query, k):
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = np.asarray(q, dtype="float32")
        scores, ids = self.index.search(q, min(k, self.count))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    @staticmethod
    def _minmax(scores):
        if not scores:
            return {}
        vals = [v for _, v in scores]
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-9:
            return {i: 1.0 for i, _ in scores}
        return {i: (s - lo) / (hi - lo) for i, s in scores}

    def _hybrid(self, query, candidate_k):
        dense = self._dense(query, candidate_k)
        lexical_raw = self.bm25.score(query)
        lexical = sorted(enumerate(lexical_raw.tolist()), key=lambda x: x[1], reverse=True)[:candidate_k]

        dn = self._minmax(dense)
        ln = self._minmax(lexical)

        ids = set(dn) | set(ln)
        merged = []
        for idx in ids:
            # Weight dense slightly higher because this is a semantic assistant.
            score = 0.65 * dn.get(idx, 0.0) + 0.35 * ln.get(idx, 0.0)
            merged.append((idx, score, dn.get(idx, 0.0), ln.get(idx, 0.0)))
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:candidate_k]

    def _get_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self.reranker

    def search(self, query, candidate_k=12, final_k=5, rerank=True):
        if self.index is None or not self.chunks:
            return []

        candidates = self._hybrid(query, candidate_k)
        if not candidates:
            return []

        if rerank:
            try:
                model = self._get_reranker()
                pairs = [[query, self.chunks[idx]["text"]] for idx, _, _, _ in candidates]
                rr = model.predict(pairs)
                order = np.argsort(-np.asarray(rr))
                chosen = []
                for j in order[:final_k]:
                    idx, hybrid_score, dense_score, lexical_score = candidates[int(j)]
                    chosen.append((idx, float(rr[int(j)]), hybrid_score, dense_score, lexical_score))
            except Exception:
                chosen = [
                    (idx, hybrid_score, hybrid_score, dense_score, lexical_score)
                    for idx, hybrid_score, dense_score, lexical_score in candidates[:final_k]
                ]
        else:
            chosen = [
                (idx, hybrid_score, hybrid_score, dense_score, lexical_score)
                for idx, hybrid_score, dense_score, lexical_score in candidates[:final_k]
            ]

        # Normalize reranker score into 0..1 for display/thresholding.
        raw = [x[1] for x in chosen]
        if raw:
            lo, hi = min(raw), max(raw)
            if abs(hi - lo) < 1e-9:
                norm = [1.0] * len(raw)
            else:
                norm = [(v - lo) / (hi - lo) for v in raw]
        else:
            norm = []

        results = []
        for row, ns in zip(chosen, norm):
            idx, score, hybrid_score, dense_score, lexical_score = row
            item = dict(self.chunks[idx])
            # Blend normalized reranker/hybrid score for a stable UI score.
            item["score"] = float(0.75 * ns + 0.25 * hybrid_score)
            item["rerank_score"] = float(score)
            item["dense_score"] = float(dense_score)
            item["lexical_score"] = float(lexical_score)
            results.append(item)
        return results
