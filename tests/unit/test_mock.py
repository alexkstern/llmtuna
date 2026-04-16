"""Tests for llmtuna.providers.mock — MockProvider."""

import pytest

from llmtuna.providers.base import Provider
from llmtuna.providers.mock import MockProvider


# ============================================================
# ABC inheritance
# ============================================================

def test_mock_provider_is_a_provider():
    """MockProvider must satisfy the Provider ABC contract — without this
    the inheritance has been silently broken."""
    p = MockProvider(responses=[])
    assert isinstance(p, Provider)


# ============================================================
# propose() — returning queued responses
# ============================================================

def test_returns_queued_response():
    p = MockProvider(responses=[{"lr": 0.001}])
    result = p.propose(system="sys", user="usr", tool_spec={"name": "t"})
    assert result == {"lr": 0.001}


def test_pops_responses_in_order():
    p = MockProvider(responses=[{"lr": 1}, {"lr": 2}, {"lr": 3}])
    assert p.propose(system="", user="", tool_spec={}) == {"lr": 1}
    assert p.propose(system="", user="", tool_spec={}) == {"lr": 2}
    assert p.propose(system="", user="", tool_spec={}) == {"lr": 3}


def test_response_dict_passes_through_unchanged():
    """Complex nested structures pass through verbatim — MockProvider
    does not interpret or transform the queued response."""
    response = {"lr": 0.001, "schedule": {"warmup": 100, "decay": "cosine"}}
    p = MockProvider(responses=[response])
    result = p.propose(system="", user="", tool_spec={})
    assert result == response


# ============================================================
# propose() — recording call arguments
# ============================================================

def test_records_call_args():
    p = MockProvider(responses=[{"x": 1}])
    p.propose(system="sys", user="usr", tool_spec={"name": "tool"})
    assert len(p.calls) == 1
    assert p.calls[0] == {
        "system": "sys",
        "user": "usr",
        "tool_spec": {"name": "tool"},
    }


def test_records_multiple_calls_in_order():
    p = MockProvider(responses=[{"x": 1}, {"x": 2}])
    p.propose(system="s1", user="u1", tool_spec={"a": 1})
    p.propose(system="s2", user="u2", tool_spec={"a": 2})
    assert len(p.calls) == 2
    assert p.calls[0]["system"] == "s1"
    assert p.calls[1]["system"] == "s2"


# ============================================================
# Empty / exhausted queue
# ============================================================

def test_raises_when_out_of_responses():
    p = MockProvider(responses=[{"x": 1}])
    p.propose(system="", user="", tool_spec={})
    with pytest.raises(IndexError, match="out of responses"):
        p.propose(system="", user="", tool_spec={})


def test_raises_immediately_on_empty_queue():
    p = MockProvider(responses=[])
    with pytest.raises(IndexError, match="out of responses"):
        p.propose(system="", user="", tool_spec={})


# ============================================================
# Construction-time isolation
# ============================================================

def test_instances_are_independent():
    p1 = MockProvider(responses=[{"x": 1}])
    p2 = MockProvider(responses=[{"x": 2}])
    assert p1.propose(system="", user="", tool_spec={}) == {"x": 1}
    assert p2.propose(system="", user="", tool_spec={}) == {"x": 2}
    assert len(p1.calls) == 1
    assert len(p2.calls) == 1


def test_responses_list_is_copied_at_construction():
    """Mutating the input list after construction must not affect the queue."""
    queue = [{"x": 1}, {"x": 2}]
    p = MockProvider(responses=queue)
    queue.append({"x": 3})
    queue.clear()
    assert p.propose(system="", user="", tool_spec={}) == {"x": 1}
    assert p.propose(system="", user="", tool_spec={}) == {"x": 2}
    with pytest.raises(IndexError):
        p.propose(system="", user="", tool_spec={})
