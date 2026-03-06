import re
import numpy as np
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunking import TextChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vector_store.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.knowledge_graph.kg_engine import KGEngine


class RAGPipeline:

    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "are",
        "was", "were", "been", "have", "has", "had", "using", "use", "used",
        "about", "what", "when", "where", "which", "their", "there", "these",
        "those", "than", "then", "also", "such", "through", "between", "they",
        "them", "its", "our", "your", "you", "how", "why", "is", "am", "be",
        "to", "of", "in", "on", "by", "at", "an", "as", "it", "or"
    }

    def __init__(self, data_folder="input/pdfs"):
        """
        Uses PDFs from input/pdfs
        """
        self.loader = PDFLoader(data_folder)
        self.chunker = TextChunker()
        self.embedding_model = EmbeddingModel()
        self.llm = LLMClient()
        # KGEngine automatically loads from JSON files
        self.kg_engine = KGEngine()

        self.vector_store = None
        self.retriever = None

        # Final packed context size after hybrid retrieval + rerank.
        self.query_top_k = 5
        self.summary_top_k = 10
        self.map_chunk_limit = 12

    @classmethod
    def _tokenize(cls, text):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", (text or "").lower())
        return [t for t in tokens if t not in cls.STOPWORDS]

    @staticmethod
    def _split_sentences(text):
        if not text:
            return []
        normalized = re.sub(r"\n+", "\n", text)
        coarse = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        sentences = []
        for part in coarse:
            part = part.strip()
            if not part:
                continue
            # Split overly long segments on semicolons to avoid giant run-on claims.
            if len(part) > 260 and ";" in part:
                subparts = [p.strip() for p in part.split(";") if p.strip()]
                sentences.extend(subparts)
            else:
                sentences.append(part)
        return sentences

    @staticmethod
    def _safe_div(num, den):
        return num / den if den else 0.0

    @staticmethod
    def _is_failed_generation(text):
        return isinstance(text, str) and text.startswith("LLM generation failed")

    @staticmethod
    def _extract_similarity(item):
        sim = item.get("similarity")
        if sim is None:
            dist = item.get("distance")
            if dist is not None:
                sim = 1.0 / (1.0 + max(float(dist), 0.0))
        if sim is None:
            return 0.0
        return max(0.0, min(1.0, float(sim)))

    @classmethod
    def _jaccard_similarity(cls, text_a, text_b):
        a = set(cls._tokenize(text_a))
        b = set(cls._tokenize(text_b))
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / len(a.union(b))

    def _rewrite_query(self, question):
        question = (question or "").strip()
        if not question:
            return question

        # LLM rewrite when available.
        if not getattr(self.llm, "offline_mode", False):
            rewrite_prompt = f"""
Rewrite the user question for retrieval over EEG emotion-recognition papers.
Keep intent unchanged, add missing domain terms only if needed, and keep it to one sentence.

Question: {question}
Rewritten query:
"""
            rewritten = self.llm.generate(rewrite_prompt, max_tokens=80)
            if isinstance(rewritten, str) and rewritten.strip() and not self._is_failed_generation(rewritten):
                return rewritten.strip().replace("\n", " ")

        # Offline heuristic rewrite.
        q = question.rstrip(" ?")
        lower = q.lower()
        if any(k in lower for k in ["lstm", "cnn", "transformer"]) and "eeg" not in lower:
            topic = q[:-4].strip() if lower.endswith(" use") else q
            return f"How is {topic} used in EEG-based emotion recognition and classification?"
        if len(self._tokenize(q)) <= 5:
            return f"{q} in EEG-based emotion recognition research papers"
        return question

    def _merge_retrieval_results(self, original_results, rewritten_results, original_query, rewritten_query):
        """
        Merge retrieval results from original + rewritten queries and keep
        best score signals per chunk id.
        """
        by_id = {}
        original_terms = set(self._tokenize(original_query))
        rewritten_terms = set(self._tokenize(rewritten_query))

        def add_items(items, source):
            for item in items:
                chunk_id = item.get("id")
                if chunk_id is None:
                    # fallback key if id is missing
                    chunk_id = f"{item.get('metadata', {}).get('paper', '')}:{hash(item.get('text', ''))}"

                text_terms = set(self._tokenize(item.get("text", "")))
                orig_overlap = self._safe_div(len(original_terms.intersection(text_terms)), len(original_terms))
                rew_overlap = self._safe_div(len(rewritten_terms.intersection(text_terms)), len(rewritten_terms))
                local_focus = max(orig_overlap, rew_overlap)

                base_score = float(item.get("hybrid_score", 0.0) or 0.0)
                # Slightly favor results obtained from original question wording.
                if source == "original":
                    base_score += 0.03

                if chunk_id not in by_id:
                    merged = dict(item)
                    merged["hybrid_score"] = base_score
                    merged["focus_score"] = local_focus
                    by_id[chunk_id] = merged
                else:
                    existing = by_id[chunk_id]
                    if base_score > float(existing.get("hybrid_score", 0.0)):
                        existing["hybrid_score"] = base_score
                    if local_focus > float(existing.get("focus_score", 0.0)):
                        existing["focus_score"] = local_focus
                    # Keep best rerank score when available.
                    if item.get("rerank_score") is not None:
                        prev_rr = existing.get("rerank_score")
                        if prev_rr is None or float(item.get("rerank_score", 0.0)) > float(prev_rr):
                            existing["rerank_score"] = float(item.get("rerank_score", 0.0))

        add_items(original_results, "original")
        add_items(rewritten_results, "rewritten")
        merged_items = list(by_id.values())
        merged_items.sort(
            key=lambda x: (
                float(x.get("focus_score", 0.0)),
                float(x.get("rerank_score", x.get("hybrid_score", 0.0))),
                float(x.get("hybrid_score", 0.0)),
            ),
            reverse=True,
        )
        return merged_items

    def _deduplicate_chunks(self, chunks, threshold=0.88):
        unique = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            duplicate = False
            for kept in unique:
                if self._jaccard_similarity(text, kept.get("text", "")) >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(chunk)
        return unique

    def _mmr_pack(self, query, chunks, k=5, diversity_lambda=0.86):
        if not chunks:
            return []
        if len(chunks) <= k:
            return chunks

        query_terms = set(self._tokenize(query))

        def relevance_score(item):
            if item.get("focus_score") is not None:
                return float(item.get("focus_score", 0.0))
            if item.get("hybrid_score") is not None:
                return float(item.get("hybrid_score", 0.0))
            sim = self._extract_similarity(item)
            text_terms = set(self._tokenize(item.get("text", "")))
            lexical = self._safe_div(len(query_terms.intersection(text_terms)), len(query_terms))
            return 0.75 * sim + 0.25 * lexical

        selected = []
        candidates = list(chunks)

        while candidates and len(selected) < k:
            best_idx = None
            best_score = float("-inf")

            for idx, cand in enumerate(candidates):
                rel = relevance_score(cand)
                if not selected:
                    mmr = rel
                else:
                    max_sim = max(
                        self._jaccard_similarity(cand.get("text", ""), sel.get("text", ""))
                        for sel in selected
                    )
                    mmr = diversity_lambda * rel - (1.0 - diversity_lambda) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(candidates.pop(best_idx))

        return selected

    def _pack_context(self, query, retrieved, k):
        deduped = self._deduplicate_chunks(retrieved)
        query_terms = set(self._tokenize(query))

        rescored = []
        for item in deduped:
            text_terms = set(self._tokenize(item.get("text", "")))
            overlap = self._safe_div(len(query_terms.intersection(text_terms)), len(query_terms))
            focus_score = max(float(item.get("focus_score", 0.0)), overlap)
            merged = dict(item)
            merged["focus_score"] = focus_score
            rescored.append(merged)

        rescored.sort(
            key=lambda x: (
                float(x.get("focus_score", 0.0)),
                float(x.get("rerank_score", x.get("hybrid_score", 0.0))),
            ),
            reverse=True,
        )

        # Remove very weakly aligned chunks unless we need them to fill context.
        filtered = [c for c in rescored if float(c.get("focus_score", 0.0)) >= 0.12]
        if len(filtered) < k:
            nonzero = [c for c in rescored if float(c.get("focus_score", 0.0)) > 0.01]
            if nonzero:
                filtered = nonzero[: max(k * 2, k)]
            else:
                filtered = rescored[: min(k, len(rescored))]

        return self._mmr_pack(query, filtered, k=k)

    def _extract_claims(self, text, max_claims=8):
        claims = self._split_sentences(text)
        filtered = [c for c in claims if len(self._tokenize(c)) >= 4 and not self._is_noisy_sentence(c)]
        return filtered[:max_claims]

    @staticmethod
    def _truncate_text(text, max_chars=240):
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _is_noisy_sentence(self, sentence):
        s = (sentence or "").strip()
        if not s:
            return True
        lowered = s.lower()
        if len(s) > 420:
            return True
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}", s):
            return True
        # Likely author/affiliation/conference metadata noise.
        noise_terms = ["department", "institute", "university", "conference", "issn", "doi", "email", "abstract-"]
        if any(term in lowered for term in noise_terms) and len(self._tokenize(s)) > 12:
            return True
        # Author-list style lines.
        if s.count(",") >= 7 and len(self._tokenize(s)) > 16:
            return True
        return False

    def _ground_answer_with_citations(self, answer, retrieved_chunks, question=None, max_claims=6):
        claims = self._extract_claims(answer, max_claims=max_claims)
        if not claims:
            return answer, {
                "claim_count": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "citation_coverage": 0.0,
                "unsupported_claim_rate": 0.0,
            }

        chunk_entries = []
        for chunk in retrieved_chunks:
            chunk_entries.append({
                "paper": chunk.get("metadata", {}).get("paper", "Unknown"),
                "tokens": set(self._tokenize(chunk.get("text", ""))),
            })

        grounded_claims = []
        supported = 0
        unsupported = 0
        found_papers = set()
        query_terms = set(self._tokenize(question or ""))

        for claim in claims:
            claim_tokens = set(self._tokenize(claim))
            if not claim_tokens:
                continue

            best_score = 0.0
            best_paper = None
            for entry in chunk_entries:
                overlap = self._safe_div(
                    len(claim_tokens.intersection(entry["tokens"])),
                    len(claim_tokens),
                )
                if overlap > best_score:
                    best_score = overlap
                    best_paper = entry["paper"]

            if best_score >= 0.25 and best_paper:
                found_papers.add(best_paper)
                supported += 1
            else:
                unsupported += 1

        # Return the original answer without citations/sources.
        grounded_answer = answer.strip()

        claim_count = max(1, supported + unsupported)

        return grounded_answer, {
            "claim_count": claim_count,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "citation_coverage": round(self._safe_div(supported, claim_count), 4),
            "unsupported_claim_rate": round(self._safe_div(unsupported, claim_count), 4),
        }

    def _rank_sentences(self, sentences, query):
        query_terms = set(self._tokenize(query))
        ranked = []
        for idx, sentence in enumerate(sentences):
            sent_terms = set(self._tokenize(sentence))
            if not sent_terms:
                continue
            overlap = self._safe_div(len(query_terms.intersection(sent_terms)), len(query_terms))
            cue_bonus = 0.0
            lowered = sentence.lower()
            if any(k in lowered for k in ["method", "model", "result", "conclusion", "dataset", "accuracy"]):
                cue_bonus = 0.25
            score = overlap + cue_bonus - (idx * 0.005)
            ranked.append((score, sentence))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _offline_answer(self, question, retrieved):
        context = " ".join([r.get("text", "") for r in retrieved if r.get("text")])
        if not context.strip():
            return "No answer generated. Unable to find relevant context in indexed papers."

        sentences = self._split_sentences(context)
        ranked = self._rank_sentences(sentences, question)
        top = [sent for _, sent in ranked[:5]]
        if not top:
            top = [context[:700]]
        return " ".join(top).strip()

    def _offline_map_chunk_summary(self, chunk_text, paper_name):
        sentences = self._split_sentences(chunk_text)
        ranked = self._rank_sentences(
            sentences,
            f"objective dataset method results limitations conclusion {paper_name}",
        )
        top = [sent for _, sent in ranked[:2]]
        return " ".join(top).strip() if top else chunk_text[:300].strip()

    def _pick_section_sentence(self, sentences, section_terms):
        best = ""
        best_score = -1.0
        for sentence in sentences:
            terms = set(self._tokenize(sentence))
            if not terms:
                continue
            score = len(terms.intersection(section_terms))
            if score > best_score:
                best_score = score
                best = sentence
        return best

    def _offline_reduce_summary(self, paper_name, map_summaries):
        combined = " ".join(map_summaries)
        sentences = self._split_sentences(combined)

        section_keywords = {
            "Objective": {"objective", "aim", "goal", "purpose", "motivation"},
            "Data/Dataset": {"dataset", "deap", "seed", "subject", "samples", "signals"},
            "Method": {"method", "approach", "model", "cnn", "lstm", "transformer", "feature"},
            "Results": {"result", "accuracy", "performance", "improved", "evaluation", "metric"},
            "Limitations": {"limitation", "challenge", "bias", "small", "future", "constraint"},
            "Conclusion": {"conclusion", "conclude", "finding", "demonstrate", "summary"},
        }

        lines = []
        for section, kws in section_keywords.items():
            sent = self._pick_section_sentence(sentences, kws)
            if not sent:
                if section == "Data/Dataset":
                    sent = f"Dataset details are not found in context for {paper_name}."
                elif section == "Limitations":
                    sent = "Limitations are not found in context."
                else:
                    sent = f"{section} is not found in context."
            lines.append(f"{section}: {sent}")

        return "\n".join(lines)

    def _map_reduce_summary(self, paper_name, retrieved_chunks, kg_results):
        selected = retrieved_chunks[: self.map_chunk_limit]
        map_summaries = []

        for chunk in selected:
            chunk_text = chunk.get("text", "")
            map_prompt = f"""
Extract concise notes from this research context with these labels:
Objective, Data/Dataset, Method, Results, Limitations, Conclusion.
Use one short line per label. If missing, write "not found in context".

Paper: {paper_name}
Context:
{chunk_text}
"""
            note = self.llm.generate(map_prompt, max_tokens=260)
            if self._is_failed_generation(note) or not isinstance(note, str) or not note.strip():
                note = self._offline_map_chunk_summary(chunk_text, paper_name)
            map_summaries.append(note.strip())

        kg_context = "\n".join(
            [f"{t['subject']} {t['relation']} {t['object']}" for t in kg_results]
        )
        reduce_prompt = f"""
Create a structured summary for '{paper_name}' using map summaries and KG facts.
Use exactly these sections:
Objective
Data/Dataset
Method
Results
Limitations
Conclusion

If any section is unsupported, write "not found in context" for that section.
Keep each section concise and evidence-based.

Map summaries:
{chr(10).join(map_summaries)}

Knowledge Graph:
{kg_context}
"""
        final_summary = self.llm.generate(reduce_prompt, max_tokens=1400)
        if self._is_failed_generation(final_summary) or not isinstance(final_summary, str) or not final_summary.strip():
            final_summary = self._offline_reduce_summary(paper_name, map_summaries)

        return final_summary.strip(), map_summaries

    # BUILD INDEX
    def build_index(self):
        papers = self.loader.load_papers()
        print(f"\nLoaded papers: {len(papers)}")

        all_chunks = []
        metadata = []

        for name, text in papers:
            if not text.strip():
                print(f"Warning: empty text in {name}")
                continue

            chunks = self.chunker.chunk_text(text)
            print(f"{name} -> {len(chunks)} chunks")

            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append(chunk)
                    metadata.append({"paper": name})

        if len(all_chunks) == 0:
            raise ValueError("No chunks created. Check PDF extraction or folder path.")

        print(f"\nCreating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks)
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dim = embeddings.shape[1]
        print("Building FAISS index...")
        self.vector_store = FAISSStore(dim)
        self.vector_store.add(embeddings, all_chunks, metadata)
        self.retriever = Retriever(self.embedding_model, self.vector_store)
        print("Index built successfully.")

    # QUERY (Hybrid + Grounded)
    def query(self, question, paper_name=None, return_context=False):
        if self.retriever is None:
            raise RuntimeError("Index is not built. Call build_index() before querying.")

        rewritten_query = self._rewrite_query(question)
        original_results = self.retriever.retrieve(question, k=max(20, self.query_top_k * 4))
        rewritten_results = self.retriever.retrieve(rewritten_query, k=max(20, self.query_top_k * 4))
        retrieved_raw = self._merge_retrieval_results(
            original_results,
            rewritten_results,
            original_query=question,
            rewritten_query=rewritten_query,
        )

        if paper_name:
            retrieved_raw = [r for r in retrieved_raw if r.get("metadata", {}).get("paper") == paper_name]

        retrieved = self._pack_context(question, retrieved_raw, self.query_top_k)
        context_vector = "\n\n".join([r.get("text", "") for r in retrieved])

        kg_results = self.kg_engine.search_kg(question, k=5)
        context_kg = "\n".join([f"{t['subject']} {t['relation']} {t['object']}" for t in kg_results])

        full_context = (
            f"Research Paper Context:\n{context_vector}\n\n"
            f"Knowledge Graph Context:\n{context_kg}"
        )

        prompt = f"""
Answer the question using only the provided context.
If a claim is not supported, explicitly say "not found in context".
You can use markdown for better readability (bold, lists).
Focus strictly on the exact user question and avoid unrelated details.
Return a comprehensive yet concise answer (4-8 sentences) directly answering the question.
Prioritize the most relevant facts first.

Original Question:
{question}

Retrieval Query:
{rewritten_query}

Research Paper Context:
{context_vector}

Knowledge Graph Context:
{context_kg}
"""

        answer = self.llm.generate(prompt)
        if self._is_failed_generation(answer) or not isinstance(answer, str) or not answer.strip():
            answer = self._offline_answer(question, retrieved)

        grounded_answer, citation_stats = self._ground_answer_with_citations(
            answer,
            retrieved,
            question=question,
            max_claims=6,
        )

        if return_context:
            return {
                "answer": grounded_answer,
                "context": full_context,
                "retrieved_chunks": retrieved,
                "kg_triples": kg_results,
                "rewritten_query": rewritten_query,
                "citation_stats": citation_stats,
            }

        return grounded_answer

    def summarize_paper(self, paper_name, return_context=False):
        """
        Summarize a specific paper using hybrid retrieval + map-reduce.
        """
        if self.retriever is None:
            raise RuntimeError("Index is not built. Call build_index() before summarizing.")

        retrieval_query = self._rewrite_query(f"summarize objectives methods results limitations {paper_name}")
        retrieved_raw = self.retriever.retrieve(retrieval_query, k=max(24, self.summary_top_k * 4))
        retrieved_raw = [r for r in retrieved_raw if r.get("metadata", {}).get("paper") == paper_name]

        # Fallback if retrieval misses the selected paper.
        if not retrieved_raw:
            papers = self.loader.load_papers()
            paper_text = next((text for name, text in papers if name == paper_name), "")
            if paper_text.strip():
                direct_chunks = self.chunker.chunk_text(paper_text)
                retrieved_raw = [
                    {
                        "id": i,
                        "text": chunk,
                        "metadata": {"paper": paper_name},
                        "distance": None,
                        "similarity": None,
                        "hybrid_score": 0.0,
                    }
                    for i, chunk in enumerate(direct_chunks[:24])
                    if chunk.strip()
                ]

        retrieved = self._pack_context(retrieval_query, retrieved_raw, self.summary_top_k)
        context_vector = "\n\n".join([r.get("text", "") for r in retrieved])

        kg_results = self.kg_engine.search_kg(paper_name, k=10)
        context_kg = "\n".join([f"{t['subject']} {t['relation']} {t['object']}" for t in kg_results])
        full_context = f"Research Paper Context:\n{context_vector}\n\nKnowledge Graph:\n{context_kg}"

        summary, map_summaries = self._map_reduce_summary(paper_name, retrieved, kg_results)

        if return_context:
            return {
                "summary": summary,
                "context": full_context,
                "retrieved_chunks": retrieved,
                "kg_triples": kg_results,
                "map_summaries": map_summaries,
            }

        return summary
