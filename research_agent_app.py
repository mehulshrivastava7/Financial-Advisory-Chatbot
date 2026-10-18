"""Standalone optional Streamlit entry point, independent of legacy UI paths.

Run from repository root:
    python -m streamlit run research_agent_app.py
"""

from agentic_advisor.streamlit_page import render_research_agent

render_research_agent()
