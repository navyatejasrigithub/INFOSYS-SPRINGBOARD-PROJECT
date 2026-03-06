# Hybrid RAG Pipeline Implementation & Evaluation

## Tasks:
- [ ] 1. Update RAGPipeline to return context alongside answer for proper evaluation
- [ ] 2. Update RAGEvaluator with comprehensive evaluation metrics
- [ ] 3. Update main.py to run pipeline with evaluation
- [ ] 4. Test the complete pipeline

## Implementation Details:
1. **RAGPipeline updates**: Add methods to return context (vector + KG) for faithfulness evaluation
2. **RAGEvaluator updates**: 
   - Add precision@k, recall@k, MRR metrics
   - Add context precision evaluation
   - Full pipeline evaluation method
3. **main.py**: Orchestrate the pipeline and evaluation
