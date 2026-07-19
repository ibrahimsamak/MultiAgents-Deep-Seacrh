"""Product search + the two LangChain tools the ProductAgent drives.

- `DuckDuckGoProductSearch` — live web search that best-effort parses prices out
  of result snippets.
- `convert_to_cad` — a LangChain `@tool` that converts a price to CAD using fixed
  exchange rates.
- `build_compare_prices_tool(products)` — factory for a LangChain `@tool` that
  picks the cheapest of the products found this turn (closes over them, so the
  model doesn't have to echo the whole list back).
"""

import re

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

# Fixed exchange rates to Canadian dollars (demo values, not live).
CAD_RATES = {
    "CAD": 1.00,
    "USD": 1.37,
    "EUR": 1.47,
    "GBP": 1.73,
    "AUD": 0.90,
}

_CURRENCY_SYMBOLS = {
    "C$": "CAD", "CA$": "CAD", "US$": "USD", "$": "USD", "€": "EUR", "£": "GBP",
}
# Match a currency symbol followed by a number, e.g. "$1,299.99" or "£249".
_PRICE_RE = re.compile(r"(C\$|CA\$|US\$|\$|€|£)\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")


def _parse_price(text):
    """Return (amount, currency) parsed from `text`, or (None, None)."""
    match = _PRICE_RE.search(text or "")
    if not match:
        return None, None
    symbol, number = match.group(1), match.group(2).replace(",", "")
    try:
        amount = float(number)
    except ValueError:
        return None, None
    return amount, _CURRENCY_SYMBOLS.get(symbol, "USD")


class DuckDuckGoProductSearch:
    """Search the web for products and return those with a parseable price."""

    def __init__(self, max_results=10):
        self.max_results = max_results

    def search(self, query):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(f"{query} price buy", max_results=self.max_results)
        except DDGSException:
            return []

        products = []
        for r in results:
            amount, currency = _parse_price(f"{r.get('title', '')} {r.get('body', '')}")
            if amount is None:
                continue
            href = r.get("href", "") or ""
            source = href.split("/")[2] if "//" in href else href
            products.append(
                {
                    "name": (r.get("title", "") or "").strip()[:100],
                    "price": amount,
                    "currency": currency,
                    "source": source,
                    "url": href,
                }
            )
        return products


@tool
def convert_to_cad(amount: float, from_currency: str) -> dict:
    """Convert a price to Canadian dollars (CAD) using fixed exchange rates.

    Args:
        amount: the price amount to convert.
        from_currency: ISO currency code, e.g. USD, EUR, GBP.
    """
    rate = CAD_RATES.get((from_currency or "").upper())
    if rate is None or not isinstance(amount, (int, float)):
        return {"error": f"unsupported currency; known: {list(CAD_RATES)}"}
    return {"amount_cad": round(amount * rate, 2), "currency": "CAD", "rate": rate}


def build_compare_prices_tool(products):
    """Return a LangChain `@tool` that picks the cheapest of `products`."""

    @tool
    def compare_prices() -> dict:
        """Compare the products found and return the single cheapest one
        (name, price, currency, source)."""
        priced = [p for p in products if isinstance(p.get("price"), (int, float))]
        if not priced:
            return {"error": "no priced products"}
        return min(priced, key=lambda p: p["price"])

    return compare_prices
