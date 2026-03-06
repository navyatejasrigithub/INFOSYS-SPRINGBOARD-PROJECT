# evaluate_model.py - Model Evaluation Script for Hybrid RAG Pipeline

"""
This script demonstrates how to run LLM evaluation and RAG pipeline evaluation.

LLM Evaluation:
- Faithfulness: Checks if answer is grounded in context
- Relevance: Checks if answer addresses the question
- Summary Quality: Evaluates summary completeness

RAG Pipeline Evaluation:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant doc
"""

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator


def run_llm_evaluation(pipeline, evaluator):
    """
    Run LLM-based evaluation (Faithfulness, Relevance, Summary Quality)
    """
    print("\n" + "=" * 60)
    print("LLM EVALUATION")
    print("=" * 60)
    
    # Test queries for LLM evaluation
    test_queries = [
        "What methods are used for EEG-based emotion recognition?",
        "How does LSTM help in emotion classification?",
        "What is the DEAP dataset used for?",
        "Explain transfer learning in EEG emotion recognition"
    ]
    
    for query in test_queries:
        print(f"\n[Query]: {query}")
        print("-" * 40)
        
        # Get answer with context
        result = pipeline.query(query, return_context=True)
        
        if isinstance(result, dict):
            answer = result.get("answer", "N/A")
            context = result.get("context", "")
            
            # Evaluate faithfulness
            print("[Evaluating Faithfulness...]")
            faithfulness = evaluator.evaluate_faithfulness(context, answer)
            print(f"Faithfulness: {faithfulness.get('score', 0.0)}/10 ({faithfulness.get('method', 'heuristic')})")
            
            # Evaluate relevance
            print("[Evaluating Relevance...]")
            relevance = evaluator.evaluate_relevance(query, answer)
            print(f"Relevance: {relevance.get('score', 0.0)}/10 ({relevance.get('method', 'heuristic')})")

            citation_metrics = evaluator.evaluate_citation_metrics(answer)
            print(
                "Citation Coverage: "
                f"{citation_metrics.get('citation_coverage', 0.0) * 100:.1f}% | "
                "Unsupported Claim Rate: "
                f"{citation_metrics.get('unsupported_claim_rate', 0.0) * 100:.1f}%"
            )
        else:
            print(f"Answer: {result}")
    
    # Evaluate summary quality
    print("\n" + "-" * 40)
    print("[Evaluating Summary Quality]")
    papers = pipeline.loader.load_papers()
    if papers:
        first_paper = papers[0][0]
        summary = pipeline.summarize_paper(first_paper)
        
        if isinstance(summary, dict):
            summary_text = summary.get("summary", summary)
        else:
            summary_text = summary
            
        summary_eval = evaluator.evaluate_summary(first_paper, summary_text)
        print(f"Summary Evaluation: {summary_eval.get('score', 0.0)}/10 ({summary_eval.get('method', 'heuristic')})")


def run_rag_pipeline_evaluation(pipeline, evaluator):
    """
    Run RAG pipeline evaluation (Precision@K, Recall@K, MRR)
    """
    print("\n" + "=" * 60)
    print("RAG PIPELINE EVALUATION (Retrieval Metrics)")
    print("=" * 60)
    
    # Test queries with ground truth (relevant paper names)
    test_queries = [
        "What methods are used for EEG-based emotion recognition?",
        "How does LSTM help in emotion classification?",
        "What is the DEAP dataset used for?",
        "Explain transfer learning in EEG emotion recognition",
        "What is the accuracy of emotion recognition using deep learning?"
    ]
    
    # Ground truth: mapping queries to relevant papers
    # These should be papers that are most relevant to each query
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
            "DEAP_A_Database_for_Emotion_Analysis_Using_Physiological_Signals.pdf",
            "Analysis of eeg signlas in the deap dataset for emotion recognition using deep learning algorithms.pdf"
        ],
        "Explain transfer learning in EEG emotion recognition": [
            "EEG-based_Emotion_Detection_Using_Unsupervised_Transfer_Learning.pdf",
            "Domain_Adaptation_Techniques_for_EEG-Based_Emotion_Recognition_A_Comparative_Study_on_Two_Public_Datasets.pdf"
        ],
        "What is the accuracy of emotion recognition using deep learning?": [
            "EEG-based_Emotion_Recognition_An_In-depth_Analysis_using_DEAP_and_SEED_Datasets.pdf",
            "Optimizing_Emotion_Classification_from_EEG_Signals_A_Comparative_Analysis_of_Optimization_Techniques_on_the_DEAP_Dataset.pdf"
        ]
    }
    
    print("\n[Running Retrieval Evaluation...]")
    results = evaluator.evaluate_pipeline_retrieval(pipeline, test_queries, ground_truth)
    
    print(f"\n[Results]")
    print(f"  Precision@5:  {results['avg_precision_at_5']:.4f}")
    print(f"  Recall@5:     {results['avg_recall_at_5']:.4f}")
    print(f"  MRR:          {results['avg_mrr']:.4f}")
    print(f"  Hit@5:        {results['avg_hit_rate_at_5']:.4f}")
    print(f"  MAP@5:        {results['avg_map_at_5']:.4f}")
    print(f"  nDCG@5:       {results['avg_ndcg_at_5']:.4f}")
    
    return results


def main():
    print("=" * 60)
    print("HYBRID RAG PIPELINE EVALUATION")
    print("=" * 60)
    
    # Initialize pipeline and evaluator
    pipeline = RAGPipeline()
    evaluator = RAGEvaluator()
    
    # Build index (loads PDFs, creates embeddings, builds FAISS)
    print("\n[1] Building Index...")
    pipeline.build_index()
    
    # Run LLM evaluation
    try:
        run_llm_evaluation(pipeline, evaluator)
    except Exception as e:
        print(f"LLM Evaluation Error: {e}")
        print("(This may be due to API quota limits)")
    
    # Run RAG pipeline evaluation
    try:
        run_rag_pipeline_evaluation(pipeline, evaluator)
    except Exception as e:
        print(f"RAG Pipeline Evaluation Error: {e}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
