# src/knowledge_graph/kg_engine.py

import json
import os
from src.llm.llm_client import LLMClient

class KGEngine:
    def __init__(self, load_from_file=True):
        self.llm = LLMClient()
        self.graph = []  # List of triples: {"subject": ..., "relation": ..., "object": ...}
        self.entities = []
        self.relationships = []
        
        # Load knowledge graph from JSON files if available
        if load_from_file:
            self.load_knowledge_graph()

    def load_knowledge_graph(self):
        """Load knowledge graph from JSON files"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Load triplets
        triplets_path = os.path.join(base_path, "triplets.json")
        if os.path.exists(triplets_path):
            try:
                with open(triplets_path, 'r') as f:
                    data = json.load(f)
                    self.graph = data.get("triplets", [])
                print(f"Loaded {len(self.graph)} triplets from triplets.json")
            except Exception as e:
                print(f"Error loading triplets.json: {e}")
        
        # Load entities
        entities_path = os.path.join(base_path, "entities.json")
        if os.path.exists(entities_path):
            try:
                with open(entities_path, 'r') as f:
                    data = json.load(f)
                    self.entities = data
                print(f"Loaded entities from entities.json")
            except Exception as e:
                print(f"Error loading entities.json: {e}")
        
        # Load relationships
        relationships_path = os.path.join(base_path, "relationships.json")
        if os.path.exists(relationships_path):
            try:
                with open(relationships_path, 'r') as f:
                    data = json.load(f)
                    self.relationships = data.get("relationships", [])
                print(f"Loaded {len(self.relationships)} relationships from relationships.json")
            except Exception as e:
                print(f"Error loading relationships.json: {e}")

    def extract_triples(self, text):
        prompt = f"Extract key entities and their relationships from the following text. Return JSON list of triples with keys subject, relation, object. Text: {text}"
        response = self.llm.generate(prompt)
        try:
            # Basic cleanup if LLM returns markdown blocks
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                triples = json.loads(json_str)
                return triples
            return []
        except Exception as e:
            print(f"Error parsing triples: {e}")
            return []

    def add_triples(self, triples):
        self.graph.extend(triples)

    def search_kg(self, query, k=5):
        query_words = set(query.lower().split())
        results = []
        for triple in self.graph:
            score = 0
            for value in triple.values():
                val_words = set(str(value).lower().split())
                score += len(query_words.intersection(val_words))
            if score > 0:
                results.append((score, triple))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:k]]
