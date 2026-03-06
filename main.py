# main.py - Hybrid RAG Pipeline with Evaluation

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator

if __name__ == "__main__":
    
    print("=" * 60)
    print("HYBRID RAG PIPELINE FOR RESEARCH PAPER SUMMARIZATION")
    print("=" * 60)
    
    # Initialize the RAG pipeline
    rag = RAGPipeline()
    
    # Build the index (loads PDFs, creates chunks, embeddings, and KG)
    print("\n[1] Building Index...")
    rag.build_index()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Test queries for evaluation
    test_queries = [
        "What methods are used for EEG-based emotion recognition?",
        "How does LSTM help in emotion classification?",
        "What is the DEAP dataset used for?",
        "Explain transfer learning in EEG emotion recognition"
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING HYBRID RAG QUERIES")
    print("=" * 60)
    
    # Run queries through the pipeline
    for query in test_queries:
        print(f"\n[Query]: {query}")
        print("-" * 40)
        
        # Get answer with context for evaluation
        result = rag.query(query, return_context=True)
        
        if isinstance(result, dict):
            print(f"[Answer]: {result.get('answer', 'N/A')}")
            print(f"[Context Chunks Retrieved]: {len(result.get('retrieved_chunks', []))}")
            print(f"[KG Triples Retrieved]: {len(result.get('kg_triples', []))}")
        else:
            print(f"[Answer]: {result}")
        
        print("-" * 40)
    
    # Summarize a specific paper
    print("\n" + "=" * 60)
    print("PAPER SUMMARIZATION")
    print("=" * 60)
    
    # Get list of papers from the loader
    papers = rag.loader.load_papers()
    if papers:
        first_paper_name = papers[0][0]
        print(f"\nSummarizing paper: {first_paper_name}")
        print("-" * 40)
        
        summary_result = rag.summarize_paper(first_paper_name, return_context=True)
        
        if isinstance(summary_result, dict):
            print(f"[Summary]: {summary_result.get('summary', 'N/A')[:500]}...")
            print(f"[Context Chunks Retrieved]: {len(summary_result.get('retrieved_chunks', []))}")
            print(f"[KG Triples Retrieved]: {len(summary_result.get('kg_triples', []))}")
            
            # Evaluate the summary
            print("\n[Evaluating Summary Quality...]")
            eval_result = evaluator.evaluate_summary(first_paper_name, summary_result.get('summary', ''))
            print(f"[Summary Evaluation]: {eval_result.get('score', 0.0)}/10 ({eval_result.get('method', 'heuristic')})")
        else:
            print(f"[Summary]: {summary_result[:500]}...")
    
    # Run LLM evaluation on a sample query
    print("\n" + "=" * 60)
    print("LLM EVALUATION")
    print("=" * 60)
    
    sample_query = "What are the key findings in EEG-based emotion recognition?"
    print(f"\n[Sample Query]: {sample_query}")
    
    eval_result = evaluator.full_evaluation(rag, sample_query)
    
    print(f"\n[Faithfulness Evaluation]:")
    faith = eval_result.get('faithfulness', {})
    print(f"{faith.get('score', 0.0)}/10 ({faith.get('method', 'heuristic')})")
    
    print(f"\n[Relevance Evaluation]:")
    rel = eval_result.get('relevance', {})
    print(f"{rel.get('score', 0.0)}/10 ({rel.get('method', 'heuristic')})")

    citation_metrics = eval_result.get("citation_metrics", {})
    print(f"\n[Citation Coverage]: {citation_metrics.get('citation_coverage', 0.0) * 100:.1f}%")
    print(f"[Unsupported Claim Rate]: {citation_metrics.get('unsupported_claim_rate', 0.0) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)
