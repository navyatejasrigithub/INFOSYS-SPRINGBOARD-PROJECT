# app.py - Streamlit UI for Hybrid RAG Pipeline

"""
Streamlit Web Application for Research Paper Query and Summarization
using Hybrid RAG Pipeline with Knowledge Graph Integration

Run with: streamlit run app.py
"""

import streamlit as st
import re
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.llm.llm_client import LLMClient

# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="ðŸ“š",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'papers_cache' not in st.session_state:
    st.session_state.papers_cache = None


IRRELEVANT_REPLY = "I can't reply because this is an irrelevant question for these research papers."


SCORING_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are",
    "was", "were", "been", "have", "has", "had", "using", "use", "used",
    "about", "what", "when", "where", "which", "their", "there", "these",
    "those", "than", "then", "also", "such", "through", "between", "they",
    "them", "its", "our", "your", "you", "how", "why", "is", "am", "be",
    "to", "of", "in", "on", "by", "at", "an", "as", "it", "or"
}

DOMAIN_KEYWORDS = {
    "eeg", "emotion", "recognition", "classifier", "classification",
    "valence", "arousal", "affective", "dataset", "deap", "seed",
    "signal", "signals", "electrode", "feature", "features",
    "brain", "bci", "neural", "network", "cnn", "lstm", "transformer",
    "deep", "learning", "physiological", "multimodal", "accuracy",
    "model", "models", "transfer", "adaptation", "subject", "cross-subject"
}

CORE_DOMAIN_KEYWORDS = {
    "eeg", "emotion", "valence", "arousal",
    "affective", "deap", "seed", "signal", "signals", "electrode",
    "bci", "cnn", "lstm", "transformer", "physiological"
}

OUT_OF_SCOPE_KEYWORDS = {
    "weather", "temperature", "rain", "sports", "match", "football", "cricket",
    "movie", "movies", "song", "songs", "recipe", "cook", "cooking", "travel",
    "hotel", "flight", "bitcoin", "crypto", "stock", "stocks", "politics",
    "president", "election", "joke", "jokes", "horoscope", "celebrity", "news",
    "capital", "prime", "minister", "history", "geography", "time", "date"
}


def _tokenize_for_score(text):
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower())
    return [t for t in tokens if t not in SCORING_STOPWORDS]


def _safe_div(num, den):
    return num / den if den else 0.0


def _clean_answer_text(answer):
    """Normalize answer text for display while preserving markdown."""
    text = (answer or "").strip()
    if not text:
        return text

    # Normalize excessive newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_similarity(item):
    sim = item.get("similarity")
    if sim is None:
        dist = item.get("distance")
        if dist is not None:
            sim = 1.0 / (1.0 + max(float(dist), 0.0))
    if sim is None:
        return None
    return max(0.0, min(1.0, float(sim)))


def _should_force_irrelevant(query, retrieved_chunks, confidence):
    """
    Final safety gate to avoid junk answers for non-domain queries.
    """
    query_terms = set(_tokenize_for_score(query))
    core_hits = len(query_terms.intersection(CORE_DOMAIN_KEYWORDS))
    if core_hits > 0:
        return False

    similarities = []
    for item in (retrieved_chunks or [])[:5]:
        sim = _extract_similarity(item)
        if sim is not None:
            similarities.append(sim)
    best_sim = max(similarities) if similarities else 0.0

    return confidence < 45.0 or best_sim < 0.50


def initialize_pipeline():
    """Initialize the RAG pipeline"""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing pipeline..."):
            st.session_state.pipeline = RAGPipeline()
            st.session_state.evaluator = RAGEvaluator()
            st.session_state.llm_client = LLMClient()
    return st.session_state.pipeline, st.session_state.evaluator, st.session_state.llm_client


def build_index():
    """Build the index (load PDFs, create embeddings, KG)"""
    pipeline, _, _ = initialize_pipeline()
    with st.spinner("Building index... This may take a few minutes."):
        try:
            pipeline.build_index()
        except Exception as e:
            st.session_state.index_built = False
            st.error(f"Failed to build index: {e}")
            return
    st.session_state.index_built = True
    st.session_state.papers_cache = None
    st.success("Index built successfully!")


def get_available_papers(pipeline, force_refresh=False):
    """Load papers once and reuse across Streamlit reruns."""
    if force_refresh or st.session_state.papers_cache is None:
        try:
            st.session_state.papers_cache = pipeline.loader.load_papers()
        except Exception as e:
            print(f"Error loading papers: {e}")
            st.session_state.papers_cache = []
    return st.session_state.papers_cache


def check_query_relevance(query, llm_client, pipeline=None):
    """Check if the query is relevant to the research papers"""
    query = (query or "").strip()
    if not query:
        return False

    query_terms = set(_tokenize_for_score(query))
    if len(query_terms) <= 2 and len(query_terms.intersection(CORE_DOMAIN_KEYWORDS)) == 0:
        return False

    core_hits = len(query_terms.intersection(CORE_DOMAIN_KEYWORDS))
    domain_hits = len(query_terms.intersection(DOMAIN_KEYWORDS))
    out_scope_hits = len(query_terms.intersection(OUT_OF_SCOPE_KEYWORDS))

    retrieval_best_sim = 0.0
    retrieval_avg_sim = 0.0
    if pipeline is not None and getattr(pipeline, "retriever", None) is not None:
        try:
            probe_results = pipeline.retriever.retrieve(query, k=3, use_rerank=False)
            sims = []
            for item in probe_results:
                sim = _extract_similarity(item)
                if sim is not None:
                    sims.append(sim)
            if sims:
                retrieval_best_sim = max(sims)
                retrieval_avg_sim = _safe_div(sum(sims), len(sims))
        except Exception as e:
            print(f"Error in relevance retrieval probe: {e}")

    # Strong out-of-scope signal
    if out_scope_hits >= 1 and core_hits == 0:
        return False

    # If no core domain anchors, require extremely strong retrieval match.
    if core_hits == 0:
        return domain_hits >= 2 and retrieval_best_sim >= 0.62

    # Strong in-scope signals
    if domain_hits >= 2 and retrieval_best_sim >= 0.28:
        return True
    if retrieval_best_sim >= 0.38 or retrieval_avg_sim >= 0.33:
        return True

    # Borderline queries: ask LLM only if available.
    if llm_client is not None and not getattr(llm_client, "offline_mode", False):
        try:
            prompt = f"""
You are a query classifier. Determine if the following question is relevant to EEG-based emotion recognition research papers.
The research papers cover topics like: EEG signals, emotion recognition, LSTM, CNN, machine learning, DEAP dataset, SEED dataset,
neural networks, signal processing, affective computing, brain-computer interfaces.

Question: {query}

Respond with ONLY one word: "RELEVANT" or "IRRELEVANT"
"""
            response = llm_client.generate(prompt, max_tokens=10)
            if response and "IRRELEVANT" in response.upper():
                return False
            if response and "RELEVANT" in response.upper():
                return True
        except Exception as e:
            print(f"Error checking query relevance with LLM: {e}")

    # Conservative fallback: if no clear evidence, mark out-of-scope.
    return False


def calculate_confidence_score(query, answer, retrieved_chunks, kg_triples):
    """Calculate confidence score using retrieval similarity and answer grounding."""
    try:
        if not retrieved_chunks:
            return 0.0

        # 1) Retrieval strength from FAISS similarity (distance-normalized)
        top_chunks = retrieved_chunks[:5]
        similarities = []
        for chunk in top_chunks:
            sim = chunk.get("similarity")
            if sim is None:
                dist = chunk.get("distance")
                if dist is not None:
                    sim = 1.0 / (1.0 + max(float(dist), 0.0))

            # Fallback if similarity info is unavailable
            if sim is None:
                text_len = len(chunk.get("text", ""))
                sim = min(text_len / 1200.0, 1.0) * 0.5

            sim = max(0.0, min(1.0, float(sim)))
            similarities.append(sim)

        retrieval_strength = _safe_div(sum(similarities), len(similarities))

        # 2) Query coverage in retrieved evidence
        query_terms = set(_tokenize_for_score(query))
        context_text = " ".join(chunk.get("text", "") for chunk in top_chunks)
        context_terms = set(_tokenize_for_score(context_text))
        query_coverage = _safe_div(len(query_terms.intersection(context_terms)), len(query_terms))

        # 3) Grounding of answer in retrieved context
        answer_clean = re.sub(r"\s*\[source:\s*[^\]]+\]", "", answer or "", flags=re.IGNORECASE)
        answer_terms = _tokenize_for_score(answer_clean)
        grounded_terms = sum(1 for tok in answer_terms if tok in context_terms)
        answer_grounding = _safe_div(grounded_terms, len(answer_terms))

        # 4) Additional support signals
        kg_support = min(len(kg_triples) / 5.0, 1.0)
        evidence_depth = min(len(retrieved_chunks) / 5.0, 1.0)

        confidence_0_to_1 = (
            0.45 * retrieval_strength
            + 0.20 * query_coverage
            + 0.20 * answer_grounding
            + 0.10 * kg_support
            + 0.05 * evidence_depth
        )

        confidence_0_to_1 = max(0.0, min(1.0, confidence_0_to_1))
        return round(confidence_0_to_1 * 100, 1)
    except Exception as e:
        print(f"Error calculating confidence score: {e}")
        return 0.0


# Sidebar
st.sidebar.title("Settings")

# Build Index Button
if not st.session_state.index_built:
    if st.sidebar.button("Build Index", type="primary"):
        build_index()
else:
    st.sidebar.success("Index Ready")

# Show available papers
pipeline, evaluator, llm_client = initialize_pipeline()

if st.session_state.index_built:
    papers = get_available_papers(pipeline)
    paper_names = [p[0] for p in papers]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Available Papers")
    if paper_names:
        for paper in paper_names:
            st.sidebar.markdown(f"- {paper[:50]}...")
    else:
        st.sidebar.info("No papers found in input/pdfs.")

# Main content - using radio for navigation
st.title("Research Paper Summarizer")
st.markdown("""
This application uses a **Hybrid RAG Pipeline** with **Knowledge Graph** integration 
to query and summarize EEG-based emotion recognition research papers.
""")

# Navigation
selected_option = st.radio(
    "Select an option:",
    ["Query Papers", "Summarize Paper", "Evaluate"],
    horizontal=True
)

# Query Papers Section
if selected_option == "Query Papers":
    st.header("Query Research Papers")
    
    if not st.session_state.index_built:
        st.warning("Please build the index first using the sidebar.")
    else:
        sample_queries = [
            "What methods are used for EEG-based emotion recognition?",
            "How does LSTM help in emotion classification?",
            "What is the DEAP dataset used for?",
            "Compare CNN and LSTM approaches for EEG emotion recognition."
        ]

        sample_placeholder = "-- Select a sample query --"
        selected_sample = st.selectbox(
            "Sample queries (optional):",
            [sample_placeholder] + sample_queries,
            key="sample_query_select"
        )

        if st.button("Use sample query", key="use_sample_query"):
            if selected_sample != sample_placeholder:
                st.session_state.query_input = selected_sample

        with st.form("query_form"):
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What methods are used for EEG emotion recognition?",
                key="query_input"
            )
            submit_query = st.form_submit_button("Search", type="primary")
        
        if submit_query and query:
            with st.spinner("Analyzing query..."):
                # First check if query is relevant
                is_relevant = check_query_relevance(query, llm_client, pipeline)
                
                if not is_relevant:
                    st.session_state.query_result = {
                        'answer': IRRELEVANT_REPLY,
                        'is_relevant': False,
                        'confidence': 0.0,
                        'relevant_chunks': [],
                        'kg_triples': []
                    }
                else:
                    # Query is relevant, proceed with RAG
                    with st.spinner("Searching..."):
                        result = pipeline.query(query, return_context=True)
                        
                        # Handle case where result might be a string instead of dict
                        if isinstance(result, dict):
                            answer = result.get('answer', 'No answer generated')
                            retrieved_chunks = result.get('retrieved_chunks', [])
                            kg_triples = result.get('kg_triples', [])
                            context = result.get('context', '')
                            rewritten_query = result.get('rewritten_query', query)
                            citation_stats = result.get('citation_stats', {})
                        else:
                            answer = str(result) if result else 'No answer generated'
                            retrieved_chunks = []
                            kg_triples = []
                            context = ''
                            rewritten_query = query
                            citation_stats = {}

                        answer = _clean_answer_text(answer)
                        
                        confidence = calculate_confidence_score(
                            query=query,
                            answer=answer,
                            retrieved_chunks=retrieved_chunks,
                            kg_triples=kg_triples
                        )

                        if _should_force_irrelevant(query, retrieved_chunks, confidence):
                            st.session_state.query_result = {
                                'answer': IRRELEVANT_REPLY,
                                'is_relevant': False,
                                'confidence': 0.0,
                                'relevant_chunks': [],
                                'kg_triples': [],
                                'context': ''
                            }
                        else:
                            st.session_state.query_result = {
                                'answer': answer,
                                'is_relevant': True,
                                'confidence': confidence,
                                'relevant_chunks': retrieved_chunks,
                                'kg_triples': kg_triples,
                                'context': context,
                                'rewritten_query': rewritten_query,
                                'citation_stats': citation_stats,
                            }
                        st.session_state.last_query = query
        
        # Display result if exists
        if st.session_state.query_result:
            result = st.session_state.query_result
            
            st.subheader("Answer:")
            
            # Show relevance status
            if not result['is_relevant']:
                st.warning(IRRELEVANT_REPLY)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Show confidence score only for relevant queries
            if result['is_relevant']:
                confidence = result.get('confidence', 0)
                rewritten_query = result.get('rewritten_query', st.session_state.last_query)
                if rewritten_query and rewritten_query != st.session_state.last_query:
                    st.caption(f"Retrieval Query: {rewritten_query}")
                
                # Color code based on confidence
                if confidence >= 70:
                    st.success(f"Confidence Score: {confidence}%")
                elif confidence >= 40:
                    st.warning(f"Confidence Score: {confidence}%")
                else:
                    st.error(f"Confidence Score: {confidence}%")
                
                # Show evaluation metrics
                col1, col2 = st.columns(2)
                with col1:
                    faithfulness_eval = evaluator.evaluate_faithfulness(
                        result.get("context", ""), 
                        result.get("answer", "")
                    )
                    faith_score = faithfulness_eval.get("score", 0.0)
                    faith_method = faithfulness_eval.get("method", "heuristic")
                    st.info(f"Faithfulness: {faith_score}/10 ({faith_method})")
                
                with col2:
                    relevance_eval = evaluator.evaluate_relevance(
                        st.session_state.last_query, 
                        result.get("answer", "")
                    )
                    rel_score = relevance_eval.get("score", 0.0)
                    rel_method = relevance_eval.get("method", "heuristic")
                    st.info(f"Relevance: {rel_score}/10 ({rel_method})")

                citation_stats = result.get("citation_stats", {}) or evaluator.evaluate_citation_metrics(result.get("answer", ""))
                citation_coverage = round(citation_stats.get("citation_coverage", 0.0) * 100, 1)
                unsupported_rate = round(citation_stats.get("unsupported_claim_rate", 0.0) * 100, 1)
                st.caption(f"Citation Coverage: {citation_coverage}% | Unsupported Claim Rate: {unsupported_rate}%")
                
                # Show context in expander
                with st.expander("View Retrieved Context"):
                    st.markdown("**Retrieved Chunks:**")
                    for i, chunk in enumerate(result.get("relevant_chunks", []), 1):
                        st.markdown(f"**Chunk {i}** (from: {chunk['metadata']['paper']})")
                        st.text(chunk['text'][:2000] + "..." if len(chunk['text']) > 2000 else chunk['text'])
                        st.markdown("---")
                    
                    st.markdown("**Knowledge Graph Triples:**")
                    kg_triples = result.get("kg_triples", [])
                    if kg_triples:
                        for triple in kg_triples:
                            st.markdown(f"- **{triple['subject']}** -> {triple['relation']} -> **{triple['object']}**")
                    else:
                        st.info("No relevant KG triples found")

# Summarize Paper Section
elif selected_option == "Summarize Paper":
    st.header("Summarize a Research Paper")
    
    if not st.session_state.index_built:
        st.warning("Please build the index first using the sidebar.")
    else:
        papers = get_available_papers(pipeline)
        paper_names = [p[0] for p in papers]

        if not paper_names:
            st.warning("No papers found in input/pdfs.")
        else:
            with st.form("summary_form"):
                selected_paper = st.selectbox(
                    "Select a paper to summarize:",
                    paper_names,
                    key="paper_select"
                )
                submit_summary = st.form_submit_button("Generate Summary", type="primary")
            
            if submit_summary:
                with st.spinner("Generating summary..."):
                    st.session_state.summary_result = pipeline.summarize_paper(selected_paper, return_context=True)
            
            # Display summary if exists
            if st.session_state.summary_result:
                result = st.session_state.summary_result
                if isinstance(result, dict):
                    st.subheader("Summary:")
                    st.markdown(f"### {result.get('summary', 'No summary generated')}")
                    
                    eval_result = evaluator.evaluate_summary(selected_paper, result.get("summary", ""))
                    score = eval_result.get("score", 0.0)
                    method = eval_result.get("method", "heuristic")
                    st.success(f"Summary Quality Score: {score}/10 ({method})")
                    
                    with st.expander("View Context Used"):
                        st.text(result.get("context", ""))

# Evaluate Section
elif selected_option == "Evaluate":
    st.header("Pipeline Evaluation")
    
    st.markdown("""
    This section evaluates the RAG pipeline using standard metrics:
    - **Precision@K**: How many retrieved docs are relevant
    - **Recall@K**: How many relevant docs were retrieved  
    - **MRR**: Mean Reciprocal Rank
    - **Hit@K**: Whether at least one relevant doc was retrieved
    - **MAP@K** and **nDCG@K**: Ranking quality of relevant results
    """)
    
    if not st.session_state.index_built:
        st.warning("Please build the index first using the sidebar.")
    else:
        test_queries = [
            "What methods are used for EEG-based emotion recognition?",
            "How does LSTM help in emotion classification?",
            "What is the DEAP dataset used for?"
        ]
        
        ground_truth = {
            "What methods are used for EEG-based emotion recognition?": [
                "EEG-Based on Emotion Recognition Using Machine Learning.pdf",
                "EmHM_A_Novel_Hybrid_Model_for_the_Emotion_Recognition_based_on_EEG_Signals.pdf"
            ],
            "How does LSTM help in emotion classification?": [
                "An_Efficient_Approach_to_EEG-Based_Emotion_Recognition_using_LSTM_Network.pdf",
                "Merged_LSTM_Model_for_emotion_classification_using_EEG_signals.pdf"
            ],
            "What is the DEAP dataset used for?": [
                "DEAP_A_Database_for_Emotion_Analysis_Using_Physiological_Signals.pdf"
            ]
        }
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                results = evaluator.evaluate_pipeline_retrieval(
                    pipeline, 
                    test_queries, 
                    ground_truth
                )
            
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision@5", f"{results['avg_precision_at_5']:.4f}")
            with col2:
                st.metric("Recall@5", f"{results['avg_recall_at_5']:.4f}")
            with col3:
                st.metric("MRR", f"{results['avg_mrr']:.4f}")

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Hit@5", f"{results['avg_hit_rate_at_5']:.4f}")
            with col5:
                st.metric("MAP@5", f"{results['avg_map_at_5']:.4f}")
            with col6:
                st.metric("nDCG@5", f"{results['avg_ndcg_at_5']:.4f}")
            
            st.subheader("Detailed Results")
            for i, query in enumerate(test_queries):
                st.markdown(f"**Query {i+1}**: {query}")
                st.markdown(f"- Precision@5: {results['detailed_results']['precision_at_k'][i]:.4f}")
                st.markdown(f"- Recall@5: {results['detailed_results']['recall_at_k'][i]:.4f}")
                st.markdown(f"- MRR: {results['detailed_results']['mrr'][i]:.4f}")
                st.markdown(f"- Hit@5: {results['detailed_results']['hit_rate_at_k'][i]:.4f}")
                st.markdown(f"- MAP@5: {results['detailed_results']['map_at_k'][i]:.4f}")
                st.markdown(f"- nDCG@5: {results['detailed_results']['ndcg_at_k'][i]:.4f}")
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
**Hybrid RAG Pipeline** with Knowledge Graph Integration
- Vector Search (FAISS) + Knowledge Graph Retrieval
- LLM-powered evaluation
- Query relevance detection
""")

