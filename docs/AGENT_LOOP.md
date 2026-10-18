# Optional read-only research agent

The code under `agentic_advisor/` adds a **bounded plan -> tool -> observe ->
plan / final** loop that does not modify the existing forecasting, risk-scoring
or allocation algorithms. It is available through an opt-in Streamlit page or
through `python -m agentic_advisor ...`.

## Why here?

The legacy UI has a ticker-analysis page and a separate news sentiment page,
but no simple way to answer iterative historical-price/volatility research
questions. A controlled loop is useful when the requested answer requires
finding evidence first, possibly running more than one tool, then forming a
plain-language summary. By contrast, the existing portfolio optimizer's fixed
numerical routine has no need for an LLM in its inner loop.

```text
User question
    |
    v
ChatCompletionsPlanner (untrusted external LLM, configured explicitly)
    |
    +-- FinalAnswer ----------------------------> answer + trace
    |
    +-- ToolCall(name, arguments)
           |
           v
    ResearchAgent checks allowlist, schema,
    step budget and duplicate call signature
           |
           v
    Registered read-only tool (yfinance)
           |
           v
    Bounded observation / error -> planner again
```

## Supported tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `price_snapshot` | `symbol` | Last available adjusted close and as-of date (not a live quote) |
| `volatility_snapshot` | `symbol`, `period` | Backward-looking annualized sample volatility of daily adjusted-close returns; `period` must be `1mo`, `3mo`, `6mo`, `1y` or `2y` |

All symbols are validated with a short ticker-character allowlist. Tool
arguments must be short strings, with no unknown keys; tool outputs must be
finite, JSON-serializable and <= 6,000 serialized characters. The agent rejects
unknown tools and repeated identical calls, catches tool errors without
exposing raw exception messages to the LLM, and stops after at most four
planner decisions by default (configurable 1 to 10). It does not accept a tool
supplied by the LLM, arbitrary URLs, script execution or brokerage credentials.

## Environment variables

Copy `.env.example` to a private `.env` if useful, and **export** the values
into the process (for example using a shell or your deployment's secret
manager). The agent deliberately does not automatically load `.env`.

| Name | Required? | Notes |
| --- | --- | --- |
| `ADVISOR_API_KEY` | yes | Secret sent to selected LLM provider |
| `ADVISOR_MODEL` | yes | Chat Completions-compatible model name |
| `ADVISOR_API_URL` | no | Defaults to `https://api.openai.com/v1/chat/completions`; HTTPS required except localhost |

Sample CLI:

```bash
export ADVISOR_API_KEY='your-key'
export ADVISOR_MODEL='your-chat-model'
python -m agentic_advisor 'What is the last adjusted close of TCS.NS and its volatility over 3mo?' --trace
```

The `--trace` flag prints tool names, arguments, results and sanitized errors;
do not enable traces for private text in public logs. In Streamlit, launch the standalone page with
`python -m streamlit run research_agent_app.py` from the repository root. An
entry can be added to `UI/app.py` separately, but this compatibility patch
leaves the legacy UI unchanged because existing forks may differ.

## Risks / limitations

- **Not autonomous portfolio management:** This agent cannot buy/sell, issue
  orders, save allocations, or access uploaded portfolio files. It must not be
  repurposed to trade without separate consent, suitability checks, audit
  logging, human review and security assessment.
- **Output quality:** The model can misinterpret observations, omit dates or
  hallucinate explanations. The UI exposes observations separately for review.
  Tool evidence is not a verified investment recommendation.
- **Data timing:** Yahoo adjusted closes are historical and may be delayed,
  revised, absent, or inconsistent across exchanges. The volatility formula
  uses `sqrt(252)` and is not predictive beta, VaR or forward volatility.
- **Privacy/cost:** The question and tool observations leave the application
  for the selected LLM provider. Do not paste financial-account identifiers,
  portfolio holdings, questionnaire answers or other personal financial data
  unless the provider's data handling has been reviewed. Every iteration may
  incur an API call and associated cost.
- **Model governance:** The system prompt asks for evidence-based, non-trading
  answers, but prompts are not enforcement. Safety-critical constraints are
  placed in the code's allowlisted tools and limits.
- **Legacy compatibility:** This is opt-in and leaves historical model and
  recommendation paths untouched. Integration of existing trained models as
  tools should follow model validation and structured-output contracts, not
  directly import scripts that train/fetch data at module import time.

## Offline tests

```bash
python -m pip install pytest
python -m pytest -q tests/test_agentic_loop.py
```

Tests stub the planner, network and market tools: observation feedback,
allowlisted tool names/args, duplicate-call stopping, max-step enforcement,
secret-safe error handling, local HTTPS validation, and basic input checks.
They do **not** establish accuracy of live Yahoo Finance or LLM responses,
nor that the entire legacy Streamlit application runs end-to-end.
