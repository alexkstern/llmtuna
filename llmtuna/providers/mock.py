"""MockProvider — test double for ``Provider`` that returns canned responses.

Use in unit tests to exercise ``Tuner`` logic without hitting a real LLM
API. Initialize with a list of dicts; each call to ``propose()`` pops the
next one and records the call args for later assertion.
"""

from llmtuna.providers.base import Provider


class MockProvider(Provider):
    """Provider that returns queued responses and records every call.

    Attributes:
        responses: Remaining queued responses; ``propose()`` pops from
            the front on each call.
        calls: Cumulative log of every ``propose()`` call's arguments,
            in order. Each entry is a dict with keys ``system``,
            ``user``, ``tool_spec``.
    """

    def __init__(self, responses: list[dict]):
        """Initialize the mock with a queue of canned responses.

        Args:
            responses: Sequence of dicts to return from successive
                ``propose()`` calls (one per call, in order). The list is
                copied so external mutations after construction do not
                affect the queue.
        """
        self.responses = list(responses)
        self.calls: list[dict] = []

    def propose(self, system: str, user: str, tool_spec: dict) -> dict:
        """Record the call and return the next queued response.

        Args:
            system: System prompt (recorded only — not used for routing).
            user: User message (recorded only).
            tool_spec: Tool definition (recorded only).

        Returns:
            The next dict from the response queue.

        Raises:
            IndexError: If the queue is empty (no more canned responses
                were provided at construction time).
        """
        self.calls.append(
            {"system": system, "user": user, "tool_spec": tool_spec}
        )
        if not self.responses:
            raise IndexError(
                f"MockProvider out of responses (call #{len(self.calls)} "
                f"but only {len(self.calls) - 1} were queued)"
            )
        return self.responses.pop(0)
