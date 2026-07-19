from .BaseAgent import BaseAgent


class SearchAgent(BaseAgent):
    """Searches the live internet for the most recent news via an injected
    search tool (see search_tools.DuckDuckGoNewsTool). If an LLM is injected,
    it answers the query from the fresh results; otherwise it returns the
    combined snippets."""

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
