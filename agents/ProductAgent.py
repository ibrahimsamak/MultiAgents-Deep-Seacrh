from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .BaseAgent import BaseAgent
from tools.product_tools import build_compare_prices_tool, convert_to_cad

SYSTEM_PROMPT = (
    "You help users shop. Call compare_prices to pick the cheapest product, then "
    "convert_to_cad to convert its price to Canadian dollars. Finish with a short "
    "answer naming the cheapest product, its original price, and its price in CAD."
)


class ProductAgent(BaseAgent):
    """Searches the web for a product, then uses a LangChain (LangGraph ReAct)
    agent to compare prices (pick the cheapest) and convert that price to CAD."""

    def __init__(self, search_tool, model="gpt-4o-mini"):
        self.search_tool = search_tool
        self.llm = ChatOpenAI(model=model, temperature=0)

    def run(self, query):
        products = self.search_tool.search(query)
        if not products:
            return "I couldn't find any products with listed prices for that search."

        listing = "\n".join(
            f"- {p['name']} — {p['price']} {p['currency']} ({p['source']})"
            for p in products
        )
        agent = create_agent(
            self.llm,
            [build_compare_prices_tool(products), convert_to_cad],
            system_prompt=SYSTEM_PROMPT,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Query: {query}\n\nProducts found:\n{listing}"}]}
        )
        return result["messages"][-1].content
