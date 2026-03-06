import os
import re
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


class CombinedResearchGraph:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    # ---------- CLEAN ----------
    def clean(self, text):
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\b\w{1,2}\b', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    # ---------- EXTRACT ----------
    def extract_papers(self):

        papers = []
        folder = Path("input/pdfs")

        for pdf in folder.glob("*.pdf"):
            try:
                with open(pdf, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)

                    text = ""
                    for p in reader.pages[:25]:
                        text += p.extract_text() or ""

                    papers.append((pdf.name, self.clean(text)))
                    print("Processed", pdf.name)
            except:
                continue

        return papers

    # ---------- BUILD GRAPH ----------
    def build_graph(self):

        papers = self.extract_papers()
        if len(papers) < 1:
            print("No papers")
            return

        names = [p[0] for p in papers]
        texts = [p[1] for p in papers]

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=300,
            ngram_range=(1,2)
        )

        tfidf = vectorizer.fit_transform(texts)
        features = vectorizer.get_feature_names_out()

        with self.driver.session(database=os.getenv("NEO4J_DATABASE","neo4j")) as session:

            session.run("MATCH (n) DETACH DELETE n")

            # Create papers
            for name in names:
                session.run("""
                MERGE (p:Paper {name:$name})
                """, name=name)

            # Paper → Keyword
            for i, name in enumerate(names):
                row = tfidf[i].toarray()[0]
                top_idx = row.argsort()[-40:]

                for idx in top_idx:
                    kw = features[idx]
                    weight = float(row[idx])

                    session.run("""
                    MERGE (k:Keyword {name:$kw})
                    WITH k
                    MATCH (p:Paper {name:$paper})
                    MERGE (p)-[r:HAS_KEYWORD]->(k)
                    SET r.weight=$w
                    """, kw=kw, paper=name, w=weight)

            # Keyword relationships
            edge_strength = defaultdict(int)

            for text in texts:
                present = [kw for kw in features if kw in text][:40]

                for i in range(len(present)):
                    for j in range(i+1,len(present)):
                        a,b = sorted([present[i],present[j]])
                        edge_strength[(a,b)] += 1

            for (a,b),s in edge_strength.items():
                session.run("""
                MATCH (a:Keyword {name:$a}),(b:Keyword {name:$b})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.strength=$s
                """, a=a,b=b,s=s)

            # Paper similarity
            sim = cosine_similarity(tfidf)

            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    if sim[i][j] > 0.35:
                        session.run("""
                        MATCH (a:Paper {name:$a}),(b:Paper {name:$b})
                        MERGE (a)-[r:SIMILAR_TO]->(b)
                        SET r.score=$s
                        """, a=names[i], b=names[j], s=float(sim[i][j]))

        print("Combined research graph ready")


if __name__ == "__main__":
    CombinedResearchGraph().build_graph()