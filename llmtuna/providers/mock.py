"""MockProvider — test double for ``Provider`` that returns canned responses.

Use in unit tests to exercise ``Tuner`` logic without hitting a real LLM
API. Initialize with a list of dicts (each in the ``Provider.propose``
return shape); each call to ``propose()`` pops the next one and records
the call args for later assertion.
"""

from llmtuna.providers.base import Provider


def _canonicalize(response: dict) -> dict:
    """Coerce a queued response dict into the full ``Provider.propose`` shape.

    For test convenience, callers may queue either a bare ``tool_args``
    dict (e.g. ``{"lr": 0.001}``) or the full response shape
    (``{"reasoning": ..., "content": ..., "tool_args": ...}``). The
    presence of a top-level ``"tool_args"`` key signals the full form;
    otherwise the dict is treated as bare ``tool_args``.

    Note: this disallows hyperparameters literally named ``tool_args`` —
    the collision is a deliberate, low-cost trade for ergonomic tests.

    Args:
        response: A dict that is either bare ``tool_args`` or the full
            ``Provider.propose`` return shape.

    Returns:
        A dict with all three keys: ``reasoning`` (str), ``content`` (str),
        ``tool_args`` (dict).
    """
    if "tool_args" in response:
        return {
            "reasoning": response.get("reasoning", ""),
            "content": response.get("content", ""),
            "tool_args": response["tool_args"],
        }
    return {"reasoning": "", "content": "", "tool_args": response}


class MockProvider(Provider):
    """Provider that returns queued responses and records every call.

    Attributes:
        responses: Remaining queued responses (in the full ``Provider.propose``
            shape after construction-time canonicalization). ``propose()``
            pops from the front on each call.
        calls: Cumulative log of every ``propose()`` call's arguments,
            in order. Each entry is a dict with keys ``system``,
            ``user``, ``tool_spec``.
    """

    def __init__(self, responses: list[dict]):
        """Initialize the mock with a queue of canned responses.

        Args:
            responses: Sequence of dicts to return from successive
                ``propose()`` calls. Each may be either a bare ``tool_args``
                dict (e.g. ``{"lr": 0.001}``) or the full response shape
                (``{"reasoning": ..., "content": ..., "tool_args": ...}``).
                Bare dicts are canonicalized at construction time. The list
                is copied so external mutations do not affect the queue.
        """
        self.responses: list[dict] = [_canonicalize(r) for r in responses]
        self.calls: list[dict] = []

    def propose(self, system: str, user: str, tool_spec: dict) -> dict:
        """Record the call and return the next queued response.

        Args:
            system: System prompt (recorded only — not used for routing).
            user: User message (recorded only).
            tool_spec: Tool definition (recorded only).

        Returns:
            The next dict from the response queue, in the full
            ``Provider.propose`` shape.

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
