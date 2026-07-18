from .BaseAgent import BaseAgent


class SearchAgent(BaseAgent):
    def __init__(self, search_tool, llm=None):
        self.search_tool = search_tool
        self.llm = llm

    def run(self, query):
        results = self.search_tool.search(query)

        if not results:
            return "No recent news found for that query."

        combined = "\n\n".join(results)

        if self.llm is None:
            return combined

        prompt = (
            "Answer the question using the recent news results below. "
            "Summarize the most relevant items and cite sources by name.\n\n"
            f"Recent news:\n{combined}\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt)
