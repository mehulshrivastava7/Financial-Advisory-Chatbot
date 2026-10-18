# Financial-Advisory-Chatbot
This repository contains code which outlines the development of an AI-powered financial advisory chatbot that provides personalized, transparent, and efficient portfolio management tailored specifically for Indian Retail Traders by integrating technologies like explainable AI and advanced natural language generation.

## Model documentation

See [Model catalogue and review notes](docs/MODELS.md) for the models used by
this research prototype, its evidence/metric gaps, model artifacts, data flow,
and known limitations. See [Agent design](docs/AGENT_LOOP.md) for the optional
read-only tool-use loop and its safety boundaries.

## Optional read-only market research agent

A **standalone Read-only Research Agent** Streamlit page is included. It is
opt-in, disabled until explicitly invoked, and has no broker/trading tools. It uses a
Chat Completions-compatible LLM endpoint and Yahoo Finance historical data.
The LLM may access your question and market observations; do not submit private
account statements or personal information to an untrusted provider.

1. Install Python dependencies for the existing app, plus `yfinance`,
   `streamlit`, and `pytest` for the optional agent/tests.
2. Copy `.env.example` to `.env`. Set `ADVISOR_API_KEY` and `ADVISOR_MODEL`.
   **Environment variables must be exported** (or loaded by your runtime);
   simply creating `.env` does not load it into this module.
3. From the repository root, run `python -m streamlit run research_agent_app.py`
   for the standalone research page. The legacy `UI/app.py` is intentionally
   unchanged to avoid conflicts with existing integrations in your fork.

For CLI use (after exporting environment variables):

```bash
python -m agentic_advisor "What is the historical volatility of RELIANCE.NS over 6mo?"
python -m pytest -q tests/test_agentic_loop.py
```

**Research only:** outputs are model-generated, may be wrong or stale, and are
not investment advice. No trading or portfolio changes are supported.
