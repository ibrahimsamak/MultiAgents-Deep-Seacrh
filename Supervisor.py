import json


class Supervisor:
    """LLM-driven supervisor agent.

    Instead of matching keywords, the supervisor reasons about the query, plans,
    and repeatedly decides which agent should run next. After each agent runs it
    observes the result and decides whether to call another agent or finish.

    `agents` is a list of specs: {"name": str, "description": str, "agent": obj}
    where `agent` implements `.run(query) -> str`.
    """

    def __init__(self, llm, agents, max_steps=4, verbose=True):
        self.llm = llm
        self.agents = {spec["name"]: spec for spec in agents}
        self.max_steps = max_steps
        self.verbose = verbose

    def route(self, query, on_event=None):
        """Plan and run agents until finished, returning the final answer.

        Each `[Supervisor]` trace line is passed to `on_event(line)` if given;
        otherwise it prints to stdout when `verbose` is set. This lets a UI
        capture the reasoning trace without changing CLI behaviour.
        """
        def emit(line):
            if on_event is not None:
                on_event(line)
            elif self.verbose:
                print(line)

        history = []
        for _ in range(self.max_steps):
            decision = self._decide(query, history)
            thought = decision.get("thought", "")
            if thought:
                emit(f"[Supervisor] {thought}")

            if decision.get("action") == "finish":
                return decision.get("answer", "")

            name = decision.get("agent")
            spec = self.agents.get(name)
            if spec is None:
                history.append(
                    {"agent": name, "result": f"(no such agent '{name}')"}
                )
                continue

            agent_input = decision.get("input") or query
            emit(f"[Supervisor] -> {name}({agent_input!r})")
            result = spec["agent"].run(agent_input)
            history.append({"agent": name, "input": agent_input, "result": result})

        # Step budget exhausted: synthesize a final answer from what we gathered.
        return self._final_answer(query, history)

    def _agent_catalog(self):
        return "\n".join(
            f"- {spec['name']}: {spec['description']}"
            for spec in self.agents.values()
        )

    def _history_text(self, history):
        if not history:
            return "(no agents have run yet)"
        lines = []
        for i, step in enumerate(history, 1):
            lines.append(
                f"{i}. agent={step['agent']} input={step.get('input')!r}\n"
                f"   result: {step['result']}"
            )
        return "\n".join(lines)

    def _decide(self, query, history):
        prompt = (
            "You are a supervisor orchestrating specialist agents to answer a "
            "user's request. Think step by step, then decide the single next "
            "action.\n\n"
            f"Available agents:\n{self._agent_catalog()}\n\n"
            f"User request: {query}\n\n"
            f"Work done so far:\n{self._history_text(history)}\n\n"
            "Decide the next action. Call an agent only if you still need more "
            "information; otherwise finish. Respond with ONLY a JSON object:\n"
            '{"thought": "<brief reasoning>", "action": "call_agent" | "finish", '
            '"agent": "<agent name if calling>", '
            '"input": "<query to send the agent, if calling>", '
            '"answer": "<final answer if finishing>"}'
        )
        return self._parse_json(self.llm.invoke(prompt))

    def _final_answer(self, query, history):
        prompt = (
            "Using the agent results below, write the best final answer to the "
            f"user's request.\n\nRequest: {query}\n\n"
            f"Agent results:\n{self._history_text(history)}\n\nFinal answer:"
        )
        return self.llm.invoke(prompt)

    @staticmethod
    def _parse_json(text):
        """Best-effort parse of a JSON object from an LLM response. Falls back to
        finishing with the raw text if no valid JSON is found."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"action": "finish", "answer": text.strip()}
