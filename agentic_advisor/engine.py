"""Small, read-only, bounded tool-use loop for financial research.

Model/planner and tool implementations are injected, so the loop can be tested
without network access or tied to a particular LLM provider.
"""

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Protocol, Sequence, Union


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, str]


@dataclass(frozen=True)
class FinalAnswer:
    text: str


Decision = Union[ToolCall, FinalAnswer]


@dataclass(frozen=True)
class Observation:
    tool: str
    arguments: Mapping[str, str]
    result: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., Mapping[str, Any]]
    required_arguments: frozenset[str]
    optional_arguments: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RunResult:
    status: str
    answer: str
    observations: tuple[Observation, ...]


class Planner(Protocol):
    def decide(
        self, question: str, observations: Sequence[Observation], tools: Sequence[Tool]
    ) -> Decision:
        """Return a read-only tool call or a final answer."""


class ResearchAgent:
    """Execute an allowlisted, bounded tool loop without making trades.

    The LLM decides which *registered* tool to request next. The agent checks
    every tool name and argument before calling it. It never evaluates code,
    shells out, performs financial transactions, or gives the LLM a way to add
    its own tools.
    """

    def __init__(
        self,
        planner: Planner,
        tools: Sequence[Tool],
        max_steps: int = 4,
        max_result_chars: int = 6000,
    ) -> None:
        if not 1 <= max_steps <= 10:
            raise ValueError("max_steps must be between 1 and 10")
        if max_result_chars < 100:
            raise ValueError("max_result_chars must be at least 100")
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("tool names must be unique")
        self.planner = planner
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.max_result_chars = max_result_chars

    def run(self, question: str) -> RunResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty")
        if len(question) > 2000:
            raise ValueError("question is too long")

        observations: list[Observation] = []
        seen_calls: set[str] = set()
        for _ in range(self.max_steps):
            try:
                decision = self.planner.decide(
                    question, tuple(observations), tuple(self.tools.values())
                )
            except Exception:
                return RunResult(
                    "planner_error",
                    "Research could not be completed because the planner failed.",
                    tuple(observations),
                )

            if isinstance(decision, FinalAnswer):
                if not decision.text.strip():
                    return RunResult(
                        "invalid_answer", "The planner returned an empty answer.",
                        tuple(observations),
                    )
                return RunResult("completed", decision.text.strip(), tuple(observations))

            if not isinstance(decision, ToolCall):
                return RunResult(
                    "invalid_decision", "The planner returned an invalid decision.",
                    tuple(observations),
                )

            tool = self.tools.get(decision.name)
            if tool is None:
                observations.append(Observation(decision.name, {}, error="tool_not_allowed"))
                # Stop rather than let the model probe arbitrary tool names.
                return RunResult("tool_rejected", "The requested tool is not allowed.", tuple(observations))

            try:
                arguments = self._validate_arguments(tool, decision.arguments)
                signature = json.dumps(
                    {"tool": decision.name, "arguments": arguments}, sort_keys=True
                )
            except (TypeError, ValueError):
                observations.append(Observation(tool.name, {}, error="invalid_arguments"))
                return RunResult("tool_rejected", "The tool arguments were rejected.", tuple(observations))

            if signature in seen_calls:
                observations.append(Observation(tool.name, arguments, error="duplicate_call"))
                return RunResult(
                    "duplicate_call", "The same tool request was repeated; research stopped.",
                    tuple(observations),
                )
            seen_calls.add(signature)

            try:
                result = tool.handler(**arguments)
                if not isinstance(result, Mapping):
                    raise ValueError("tool result must be an object")
                serialized = json.dumps(result, allow_nan=False)
                if len(serialized) > self.max_result_chars:
                    raise ValueError("tool result exceeds size limit")
                observations.append(Observation(tool.name, arguments, result=dict(result)))
            except Exception as exc:
                # Deliberately do not forward raw exceptions: third-party APIs
                # sometimes include sensitive request information in messages.
                observations.append(
                    Observation(tool.name, arguments, error=type(exc).__name__)
                )

        return RunResult(
            "step_limit",
            "Research stopped at the configured step limit. No conclusion was generated.",
            tuple(observations),
        )

    @staticmethod
    def _validate_arguments(tool: Tool, arguments: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be an object")
        required = tool.required_arguments
        allowed = required | tool.optional_arguments
        if not required <= arguments.keys() or not arguments.keys() <= allowed:
            raise ValueError("missing or unsupported argument")
        cleaned = {}
        for key, value in arguments.items():
            if not isinstance(value, str) or not value.strip() or len(value) > 64:
                raise ValueError("arguments must be non-empty short strings")
            cleaned[key] = value.strip()
        return cleaned
