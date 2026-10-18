"""Run the optional read-only research agent from a terminal.

Usage:
  ADVISOR_API_KEY=... ADVISOR_MODEL=... python -m agentic_advisor \
      'What is the historical volatility of RELIANCE.NS?'
"""

import argparse
import json
import os
import sys

from .engine import ResearchAgent
from .planner import ChatCompletionsPlanner
from .tools import market_tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only financial research agent")
    parser.add_argument("question", help="Research question, quoted as one argument")
    parser.add_argument("--max-steps", type=int, default=4, help="Planner decisions, 1 to 10")
    parser.add_argument("--trace", action="store_true", help="Print tool observations")
    args = parser.parse_args()

    api_key = os.environ.get("ADVISOR_API_KEY")
    model = os.environ.get("ADVISOR_MODEL")
    endpoint = os.environ.get(
        "ADVISOR_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    if not api_key or not model or api_key.startswith("replace-with-") or model.startswith("replace-with-"):
        parser.error("Set real ADVISOR_API_KEY and ADVISOR_MODEL environment values")

    planner = ChatCompletionsPlanner(api_key, model, endpoint)
    agent = ResearchAgent(planner, market_tools(), max_steps=args.max_steps)
    result = agent.run(args.question)
    print(result.answer)
    if args.trace:
        print(json.dumps({"status": result.status, "observations": [
            {"tool": obs.tool, "arguments": dict(obs.arguments),
             "result": obs.result, "error": obs.error}
            for obs in result.observations
        ]}, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
