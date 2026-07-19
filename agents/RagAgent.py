
from .BaseAgent import BaseAgent

class RagAgent(BaseAgent):
    def __init__(self, vector_db, llm=None):
        self.vector_db = vector_db
        self.llm = llm

    def search(self, query):
        # Backed by Chroma (see knowledge_base.build_news_vector_db).
        docs = self.vector_db.similarity_search(query, k=3)
        return docs

    def summarize(self, query, docs):
        context = "\n".join([doc.page_content for doc in docs])

        if self.llm is None:
            # No LLM injected: fall back to returning the raw context.
            return f"Summary:\n{context[:500]}..."

        prompt = (
            "Answer the question using only the news articles below. "
            "If the articles do not contain the answer, say so.\n\n"
            f"News articles:\n{context}\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt)

    def run(self, query):
        docs = self.search(query)
        return self.summarize(query, docs)