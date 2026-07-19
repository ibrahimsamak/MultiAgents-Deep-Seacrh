from openai import OpenAI


class OpenAILLM:
    """Minimal LLM backend exposing the `.invoke(query)` interface expected by
    LLMAgent and RagAgent. Wraps the OpenAI Chat Completions API."""

    def __init__(self, model="gpt-4o-mini", client=None):
        self.model = model
        self.client = client or OpenAI()

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
