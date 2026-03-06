# src/retrieval/retriever.py

import math
import re
from collections import Counter, defaultdict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class Retriever:
    """
    Hybrid retriever:
    - Dense retrieval (FAISS)
    - Lexical retrieval (BM25-like)
    - Optional cross-encoder reranking
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        dense_weight=0.65,
        lexical_weight=0.35,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_enabled=True,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rerank_enabled = rerank_enabled
        self.rerank_model_name = rerank_model

        self._doc_tokens = {}
        self._doc_len = {}
        self._df = defaultdict(int)
        self._idf = {}
        self._avg_doc_len = 1.0
        self._k1 = 1.5
        self._b = 0.75

        self._reranker = None
        self._build_lexical_index()

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", (text or "").lower())

    def _build_lexical_index(self):
        chunks = getattr(self.vector_store, "text_chunks", [])
        if not chunks:
            return

        total_len = 0
        for idx, text in enumerate(chunks):
            tokens = self._tokenize(text)
            self._doc_tokens[idx] = tokens
            doc_len = len(tokens)
            self._doc_len[idx] = doc_len
            total_len += doc_len

            for term in set(tokens):
                self._df[term] += 1

        n_docs = len(chunks)
        self._avg_doc_len = (total_len / n_docs) if n_docs else 1.0

        for term, df in self._df.items():
            # Standard BM25 idf with +1 stabilization.
            self._idf[term] = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))

    def _lexical_search(self, query, top_n):
        query_terms = self._tokenize(query)
        if not query_terms or not self._doc_tokens:
            return []

        scores = {}
        for doc_id, tokens in self._doc_tokens.items():
            if not tokens:
                continue
            tf = Counter(tokens)
            doc_len = self._doc_len.get(doc_id, 0)
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * (doc_len / self._avg_doc_len))
                if denom <= 0:
                    continue
                score += idf * ((freq * (self._k1 + 1)) / denom)

            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for doc_id, score in ranked:
            results.append({
                "id": int(doc_id),
                "text": self.vector_store.text_chunks[doc_id],
                "metadata": self.vector_store.metadata[doc_id],
                "distance": None,
                "similarity": None,
                "lexical_score": float(score),
            })
        return results

    @staticmethod
    def _min_max_normalize(values):
        if not values:
            return {}
        v_min = min(values)
        v_max = max(values)
        if v_max - v_min < 1e-9:
            return {idx: 1.0 for idx in range(len(values))}
        return {
            idx: (v - v_min) / (v_max - v_min)
            for idx, v in enumerate(values)
        }

    def _get_or_create_reranker(self):
        if not self.rerank_enabled:
            return None
        if self._reranker is not None:
            return self._reranker
        if CrossEncoder is None:
            return None
        try:
            self._reranker = CrossEncoder(self.rerank_model_name)
            return self._reranker
        except Exception as e:
            print(f"Reranker unavailable ({self.rerank_model_name}): {e}")
            return None

    def _rerank(self, query, candidates):
        reranker = self._get_or_create_reranker()
        if reranker is None or not candidates:
            return candidates

        try:
            pairs = [[query, item["text"]] for item in candidates]
            scores = reranker.predict(pairs)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
            return candidates
        except Exception as e:
            print(f"Reranking failed, using hybrid order: {e}")
            return candidates

    def retrieve(self, query, k=5, use_rerank=True):
        # Pull broader pool for reranking, smaller pool for fast mode.
        candidate_k = max(20, k * 4) if use_rerank else max(12, k * 2)

        query_embedding = self.embedding_model.encode([query])
        dense_results = self.vector_store.search(query_embedding, candidate_k)
        lexical_results = self._lexical_search(query, candidate_k)

        by_id = {}

        for item in dense_results:
            doc_id = item.get("id")
            if doc_id is None:
                continue
            by_id[doc_id] = {
                **item,
                "dense_score": float(item.get("similarity", 0.0) or 0.0),
                "lexical_score": 0.0,
            }

        for item in lexical_results:
            doc_id = item["id"]
            if doc_id in by_id:
                by_id[doc_id]["lexical_score"] = float(item.get("lexical_score", 0.0))
            else:
                by_id[doc_id] = {
                    **item,
                    "dense_score": 0.0,
                    "lexical_score": float(item.get("lexical_score", 0.0)),
                }

        if not by_id:
            return []

        items = list(by_id.values())
        dense_norm_map = self._min_max_normalize([it.get("dense_score", 0.0) for it in items])
        lexical_norm_map = self._min_max_normalize([it.get("lexical_score", 0.0) for it in items])

        for idx, item in enumerate(items):
            dense_n = dense_norm_map.get(idx, 0.0)
            lex_n = lexical_norm_map.get(idx, 0.0)
            item["hybrid_score"] = (self.dense_weight * dense_n) + (self.lexical_weight * lex_n)

        items.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

        if use_rerank:
            rerank_pool = items[:candidate_k]
            reranked = self._rerank(query, rerank_pool)
            return reranked[:k]

        return items[:k]
