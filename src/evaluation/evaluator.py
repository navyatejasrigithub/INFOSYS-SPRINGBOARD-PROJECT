# src/evaluation/evaluator.py

import math
import re
from src.llm.llm_client import LLMClient


class RAGEvaluator:
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "are",
        "was", "were", "been", "have", "has", "had", "using", "use", "used",
        "about", "what", "when", "where", "which", "their", "there", "these",
        "those", "than", "then", "also", "such", "through", "between", "they",
        "them", "its", "our", "your", "you", "how", "why", "can", "could",
        "should", "would", "will", "may", "might", "must", "is", "am", "be",
        "to", "of", "in", "on", "by", "at", "an", "as", "it", "or"
    }

    def __init__(self):
        self.llm = LLMClient()

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, value))

    @staticmethod
    def _safe_div(num, den):
        return num / den if den else 0.0

    @classmethod
    def _tokenize(cls, text):
        if not text:
            return []
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower())
        return [t for t in tokens if t not in cls.STOPWORDS]

    @staticmethod
    def _split_sentences(text):
        if not text:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _strip_source_tags(text):
        if not text:
            return ""
        return re.sub(r"\s*\[source:\s*[^\]]+\]", "", text, flags=re.IGNORECASE).strip()

    def _parse_llm_score(self, response):
        if not isinstance(response, str):
            return None
        if not response.strip() or response.startswith("LLM generation failed"):
            return None

        score_match = re.search(r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", response, flags=re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", response)

        if not score_match:
            return None

        score = self._clamp(float(score_match.group(1)), 0.0, 10.0)
        return {
            "score": round(score, 2),
            "reason": response.strip(),
            "method": "llm"
        }

    def _heuristic_faithfulness(self, context, answer):
        answer = self._strip_source_tags(answer)
        context_tokens = set(self._tokenize(context))
        answer_tokens = self._tokenize(answer)

        if not answer_tokens:
            return {
                "score": 0.0,
                "reason": "Answer has insufficient content to evaluate faithfulness.",
                "method": "heuristic"
            }

        token_coverage = self._safe_div(
            sum(1 for tok in answer_tokens if tok in context_tokens),
            len(answer_tokens)
        )

        context_sentences = self._split_sentences(context)
        context_sentence_tokens = [set(self._tokenize(s)) for s in context_sentences if len(self._tokenize(s)) >= 4]

        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            answer_sentences = [answer]

        supported = 0
        for sentence in answer_sentences:
            sent_tokens = set(self._tokenize(sentence))
            if not sent_tokens:
                continue

            best_support = 0.0
            for ctx_tokens in context_sentence_tokens:
                overlap = self._safe_div(len(sent_tokens.intersection(ctx_tokens)), len(sent_tokens))
                if overlap > best_support:
                    best_support = overlap

            if best_support >= 0.35:
                supported += 1

        sentence_support = self._safe_div(supported, len(answer_sentences))
        score = (0.65 * token_coverage + 0.35 * sentence_support) * 10.0
        score = round(self._clamp(score, 0.0, 10.0), 2)

        return {
            "score": score,
            "reason": (
                f"Heuristic faithfulness based on answer-token support in context "
                f"({token_coverage:.2f}) and sentence grounding ({sentence_support:.2f})."
            ),
            "method": "heuristic"
        }

    def _heuristic_relevance(self, question, answer):
        answer = self._strip_source_tags(answer)
        question_terms = set(self._tokenize(question))
        answer_terms = set(self._tokenize(answer))

        if not question_terms or not answer_terms:
            return {
                "score": 0.0,
                "reason": "Question or answer has insufficient lexical content for relevance scoring.",
                "method": "heuristic"
            }

        overlap = question_terms.intersection(answer_terms)
        query_coverage = self._safe_div(len(overlap), len(question_terms))
        answer_focus = self._safe_div(len(overlap), len(answer_terms))
        focus_scaled = self._clamp(answer_focus * 3.0, 0.0, 1.0)
        length_adequacy = self._clamp(len(self._tokenize(answer)) / 45.0, 0.0, 1.0)

        score = (0.65 * query_coverage + 0.20 * focus_scaled + 0.15 * length_adequacy) * 10.0
        score = round(self._clamp(score, 0.0, 10.0), 2)

        return {
            "score": score,
            "reason": (
                f"Heuristic relevance based on query coverage ({query_coverage:.2f}), "
                f"answer focus ({answer_focus:.2f}), and length adequacy ({length_adequacy:.2f})."
            ),
            "method": "heuristic"
        }

    def _heuristic_summary(self, paper_name, summary):
        summary_tokens = self._tokenize(summary)
        token_count = len(summary_tokens)

        if token_count == 0:
            return {
                "score": 0.0,
                "reason": "Summary is empty or too short to evaluate.",
                "method": "heuristic"
            }

        # Prefer summaries in a practical range for this project.
        min_tokens = 80
        max_tokens = 380
        if token_count < min_tokens:
            length_score = self._clamp(token_count / min_tokens, 0.0, 1.0)
        elif token_count > max_tokens:
            length_score = self._clamp(max_tokens / token_count, 0.3, 1.0)
        else:
            length_score = 1.0

        token_set = set(summary_tokens)
        section_groups = {
            "objective": {"objective", "aim", "goal", "motivation", "purpose"},
            "method": {"method", "approach", "model", "framework", "dataset", "feature", "training"},
            "result": {"result", "accuracy", "performance", "improve", "evaluation", "metric"},
            "conclusion": {"conclusion", "finding", "future", "limitation", "demonstrate"},
        }
        covered_sections = sum(1 for words in section_groups.values() if token_set.intersection(words))
        section_coverage = self._safe_div(covered_sections, len(section_groups))

        name_terms = set(self._tokenize(paper_name))
        name_overlap = self._safe_div(len(token_set.intersection(name_terms)), len(name_terms))

        sentence_count = len(self._split_sentences(summary))
        coherence = self._clamp(sentence_count / 6.0, 0.0, 1.0)

        score = (
            0.40 * section_coverage
            + 0.35 * length_score
            + 0.15 * name_overlap
            + 0.10 * coherence
        ) * 10.0
        score = round(self._clamp(score, 0.0, 10.0), 2)

        return {
            "score": score,
            "reason": (
                f"Heuristic summary quality from section coverage ({section_coverage:.2f}), "
                f"length quality ({length_score:.2f}), paper alignment ({name_overlap:.2f}), "
                f"and structure ({coherence:.2f})."
            ),
            "method": "heuristic"
        }

    def evaluate_faithfulness(self, context, answer):
        """
        Evaluate if the answer is faithful to the context.
        Returns dict: {"score": 0-10, "reason": str, "method": "llm|heuristic"}.
        """
        answer_clean = self._strip_source_tags(answer)
        prompt = f"""
        Evaluate if the answer is faithful to the provided context.
        Score from 0 to 10 and explain briefly.

        Context:
        {context}

        Answer:
        {answer_clean}

        Output format: Score: [score], Reason: [reason]
        """
        response = self.llm.generate(prompt)
        parsed = self._parse_llm_score(response)
        if parsed:
            return parsed
        return self._heuristic_faithfulness(context, answer_clean)

    def evaluate_relevance(self, question, answer):
        """
        Evaluate if the answer is relevant to the question.
        Returns dict: {"score": 0-10, "reason": str, "method": "llm|heuristic"}.
        """
        answer_clean = self._strip_source_tags(answer)
        prompt = f"""
        Evaluate if the answer is relevant to the question.
        Score from 0 to 10 and explain briefly.

        Question:
        {question}

        Answer:
        {answer_clean}

        Output format: Score: [score], Reason: [reason]
        """
        response = self.llm.generate(prompt)
        parsed = self._parse_llm_score(response)
        if parsed:
            return parsed
        return self._heuristic_relevance(question, answer_clean)

    def evaluate_citation_metrics(self, answer):
        """
        Compute citation coverage and unsupported-claim rate from answer source tags.
        Expected tag format: [source: ...]
        """
        claims = [s for s in self._split_sentences(answer) if len(self._tokenize(self._strip_source_tags(s))) >= 4]
        claim_count = len(claims)
        if claim_count == 0:
            return {
                "claim_count": 0,
                "cited_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "citation_coverage": 0.0,
                "unsupported_claim_rate": 0.0,
                "method": "heuristic",
            }

        cited = 0
        supported = 0
        unsupported = 0
        for claim in claims:
            match = re.search(r"\[source:\s*([^\]]+)\]", claim, flags=re.IGNORECASE)
            if not match:
                continue
            cited += 1
            source_text = match.group(1).strip().lower()
            if "not found in context" in source_text:
                unsupported += 1
            else:
                supported += 1

        return {
            "claim_count": claim_count,
            "cited_claims": cited,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "citation_coverage": round(self._safe_div(cited, claim_count), 4),
            "unsupported_claim_rate": round(self._safe_div(unsupported, claim_count), 4),
            "method": "heuristic",
        }

    def evaluate_summary(self, paper_name, summary):
        """
        Evaluate summary quality.
        Returns dict: {"score": 0-10, "reason": str, "method": "llm|heuristic"}.
        """
        prompt = f"""
        Evaluate the summary quality for paper '{paper_name}'.
        Check objectives, methodology, key results, and conclusions.
        Score from 0 to 10 and explain briefly.

        Summary:
        {summary}

        Output format: Score: [score], Feedback: [feedback]
        """
        response = self.llm.generate(prompt)
        parsed = self._parse_llm_score(response)
        if parsed:
            return parsed
        return self._heuristic_summary(paper_name, summary)

    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k=5):
        retrieved_at_k = retrieved_docs[:k]
        if not retrieved_at_k:
            return 0.0
        relevant_set = set(relevant_docs)
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        return self._safe_div(relevant_retrieved, len(retrieved_at_k))

    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k=5):
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        if not relevant_set:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        return self._safe_div(relevant_retrieved, len(relevant_set))

    def calculate_mrr(self, retrieved_docs, relevant_docs):
        relevant_set = set(relevant_docs)
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                return 1.0 / i
        return 0.0

    def calculate_hit_rate_at_k(self, retrieved_docs, relevant_docs, k=5):
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        return 1.0 if any(doc in relevant_set for doc in retrieved_at_k) else 0.0

    def calculate_average_precision_at_k(self, retrieved_docs, relevant_docs, k=5):
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        if not relevant_set:
            return 0.0

        hit_count = 0
        precision_sum = 0.0
        for i, doc in enumerate(retrieved_at_k, 1):
            if doc in relevant_set:
                hit_count += 1
                precision_sum += hit_count / i

        denom = min(len(relevant_set), k)
        return self._safe_div(precision_sum, denom)

    def calculate_ndcg_at_k(self, retrieved_docs, relevant_docs, k=5):
        retrieved_at_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        if not relevant_set:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(retrieved_at_k, 1):
            rel_i = 1.0 if doc in relevant_set else 0.0
            if rel_i:
                dcg += rel_i / math.log2(i + 1)

        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        return self._safe_div(dcg, idcg)

    def full_evaluation(self, pipeline, question, paper_name=None):
        """
        Run full evaluation for one query.
        """
        result = pipeline.query(question, paper_name, return_context=True)

        answer = result.get("answer", result) if isinstance(result, dict) else result
        context = result.get("context", "") if isinstance(result, dict) else ""
        retrieved_chunks = result.get("retrieved_chunks", []) if isinstance(result, dict) else []

        faithfulness = self.evaluate_faithfulness(context, answer)
        relevance = self.evaluate_relevance(question, answer)
        citation_metrics = self.evaluate_citation_metrics(answer)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_chunks,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "citation_metrics": citation_metrics,
        }

    def evaluate_pipeline_retrieval(self, pipeline, test_queries, ground_truth):
        """
        Evaluate retrieval quality with ranking metrics.
        """
        results = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],
            "hit_rate_at_k": [],
            "map_at_k": [],
            "ndcg_at_k": []
        }

        for query in test_queries:
            retrieved = pipeline.retriever.retrieve(query, k=5)
            retrieved_papers = [r["metadata"]["paper"] for r in retrieved]
            relevant_papers = ground_truth.get(query, [])

            precision = self.calculate_precision_at_k(retrieved_papers, relevant_papers, k=5)
            recall = self.calculate_recall_at_k(retrieved_papers, relevant_papers, k=5)
            mrr = self.calculate_mrr(retrieved_papers, relevant_papers)
            hit_rate = self.calculate_hit_rate_at_k(retrieved_papers, relevant_papers, k=5)
            map_at_5 = self.calculate_average_precision_at_k(retrieved_papers, relevant_papers, k=5)
            ndcg = self.calculate_ndcg_at_k(retrieved_papers, relevant_papers, k=5)

            results["precision_at_k"].append(precision)
            results["recall_at_k"].append(recall)
            results["mrr"].append(mrr)
            results["hit_rate_at_k"].append(hit_rate)
            results["map_at_k"].append(map_at_5)
            results["ndcg_at_k"].append(ndcg)

        avg_precision = self._safe_div(sum(results["precision_at_k"]), len(results["precision_at_k"]))
        avg_recall = self._safe_div(sum(results["recall_at_k"]), len(results["recall_at_k"]))
        avg_mrr = self._safe_div(sum(results["mrr"]), len(results["mrr"]))
        avg_hit_rate = self._safe_div(sum(results["hit_rate_at_k"]), len(results["hit_rate_at_k"]))
        avg_map = self._safe_div(sum(results["map_at_k"]), len(results["map_at_k"]))
        avg_ndcg = self._safe_div(sum(results["ndcg_at_k"]), len(results["ndcg_at_k"]))

        return {
            "avg_precision_at_5": avg_precision,
            "avg_recall_at_5": avg_recall,
            "avg_mrr": avg_mrr,
            "avg_hit_rate_at_5": avg_hit_rate,
            "avg_map_at_5": avg_map,
            "avg_ndcg_at_5": avg_ndcg,
            "detailed_results": results
        }
