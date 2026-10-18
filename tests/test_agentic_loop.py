"""Offline tests: no network, model, broker, or training artifacts required."""

import json
from types import SimpleNamespace

import pytest

from agentic_advisor import FinalAnswer, ResearchAgent, Tool, ToolCall
from agentic_advisor.planner import ChatCompletionsPlanner
from agentic_advisor.tools import _valid_symbol, market_tools


class SequentialPlanner:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.seen = []

    def decide(self, question, observations, tools):
        self.seen.append(tuple(observations))
        return next(self.decisions)


def fake_tool():
    return Tool(
        name="sample",
        description="Read-only sample",
        handler=lambda symbol: {"symbol": symbol, "value": 42},
        required_arguments=frozenset({"symbol"}),
    )


def test_tool_observation_is_passed_back_for_second_decision():
    planner = SequentialPlanner(ToolCall("sample", {"symbol": "TCS.NS"}), FinalAnswer("Observed 42."))
    result = ResearchAgent(planner, [fake_tool()]).run("Check TCS.NS")
    assert result.status == "completed"
    assert result.answer == "Observed 42."
    assert planner.seen[1][0].result == {"symbol": "TCS.NS", "value": 42}


def test_unknown_tool_cannot_execute():
    planner = SequentialPlanner(ToolCall("execute_trade", {"symbol": "TCS.NS"}))
    result = ResearchAgent(planner, [fake_tool()]).run("Buy TCS")
    assert result.status == "tool_rejected"
    assert result.observations[0].error == "tool_not_allowed"


def test_extra_arguments_cannot_reach_handler():
    planner = SequentialPlanner(ToolCall("sample", {"symbol": "TCS.NS", "quantity": "100"}))
    result = ResearchAgent(planner, [fake_tool()]).run("Check TCS")
    assert result.status == "tool_rejected"
    assert result.observations[0].error == "invalid_arguments"


def test_duplicate_tool_call_stops_loop():
    call = ToolCall("sample", {"symbol": "TCS.NS"})
    result = ResearchAgent(SequentialPlanner(call, call), [fake_tool()]).run("Check TCS")
    assert result.status == "duplicate_call"
    assert len(result.observations) == 2


def test_step_budget_is_enforced():
    planner = SequentialPlanner(ToolCall("sample", {"symbol": "TCS.NS"}), FinalAnswer("done"))
    result = ResearchAgent(planner, [fake_tool()], max_steps=1).run("Check TCS")
    assert result.status == "step_limit"
    assert "No conclusion" in result.answer


def test_tool_failure_is_observable_and_recoverable():
    def fail(symbol):
        raise RuntimeError("SECRET-DO-NOT-EXPOSE")
    tool = Tool("sample", "sample", fail, frozenset({"symbol"}))
    planner = SequentialPlanner(ToolCall("sample", {"symbol": "TCS.NS"}), FinalAnswer("No data available."))
    result = ResearchAgent(planner, [tool]).run("Check TCS")
    assert result.status == "completed"
    assert result.observations[0].error == "RuntimeError"
    assert "SECRET" not in str(result)


def test_invalid_symbol_rejected_before_yfinance_import():
    with pytest.raises(ValueError):
        _valid_symbol("AAPL; rm -rf /")
    assert _valid_symbol("reliance.ns") == "RELIANCE.NS"
    assert {tool.name for tool in market_tools()} == {"price_snapshot", "volatility_snapshot"}


def test_planner_parses_compliant_json_without_network(monkeypatch):
    result = {"choices": [{"message": {"content": json.dumps({
        "action": "tool", "tool": "sample", "arguments": {"symbol": "TCS.NS"}
    })}}]}
    class Response:
        def __enter__(self):
            return SimpleNamespace(read=lambda n: json.dumps(result).encode())
        def __exit__(self, *args):
            return None
    def fake_open(req, timeout):
        assert req.full_url == "https://example.test/v1/chat/completions"
        assert timeout == 20
        assert b"sample" in req.data
        return Response()
    monkeypatch.setattr("agentic_advisor.planner.urlopen", fake_open)
    planner = ChatCompletionsPlanner("test-key", "test-model", "https://example.test/v1/chat/completions")
    action = planner.decide("Research", [], [fake_tool()])
    assert action == ToolCall("sample", {"symbol": "TCS.NS"})


def test_insecure_remote_endpoint_rejected():
    with pytest.raises(ValueError):
        ChatCompletionsPlanner("key", "model", "http://example.com/api")


def test_empty_and_overlong_questions_rejected():
    agent = ResearchAgent(SequentialPlanner(FinalAnswer("ok")), [fake_tool()])
    with pytest.raises(ValueError):
        agent.run("")
    with pytest.raises(ValueError):
        agent.run("a" * 2001)


def test_snapshot_tools_return_dated_evidence_without_live_price_claims(monkeypatch):
    # In-memory history isolates this test from the network and broker APIs.
    import sys
    import pandas as pd
    from agentic_advisor.tools import price_snapshot, volatility_snapshot

    index = pd.date_range('2024-01-01', periods=4)
    history = pd.DataFrame({'Close': [100.0, 101.0, 102.0, 103.0]}, index=index)

    class FakeTicker:
        def history(self, period, auto_adjust):
            assert auto_adjust is True
            assert period in {'1mo', '3mo'}
            return history

    monkeypatch.setitem(sys.modules, 'yfinance', SimpleNamespace(Ticker=lambda sym: FakeTicker()))
    price = price_snapshot('TCS.NS')
    assert price['adjusted_close'] == 103.0
    assert price['as_of'] == '2024-01-04'
    assert 'not a live' in price['note']
    vol = volatility_snapshot('TCS.NS', '3mo')
    assert vol['observations'] == 3
    assert vol['annualized_realized_volatility'] >= 0
    assert 'Historical' in vol['note']


def test_period_allowlist_rejects_bad_requests_before_network():
    from agentic_advisor.tools import _history
    with pytest.raises(ValueError):
        _history('TCS.NS', 'max')


def test_tool_result_size_budget_prevents_unbounded_prompt_growth():
    oversized = Tool('large', 'large', lambda symbol: {'text': 'A' * 7000},
                     frozenset({'symbol'}))
    planner = SequentialPlanner(ToolCall('large', {'symbol': 'TCS.NS'}),
                                FinalAnswer('Data unavailable.'))
    result = ResearchAgent(planner, [oversized]).run('Explain')
    assert result.status == 'completed'
    assert result.observations[0].error == 'ValueError'
    assert result.observations[0].result is None
