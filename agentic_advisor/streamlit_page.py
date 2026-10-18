"""Optional research-agent panel; imported only when its Streamlit page is selected."""

import os


def render_research_agent() -> None:
    """Run a bounded read-only analysis only after the user explicitly clicks Run."""
    import streamlit as st

    from .engine import ResearchAgent
    from .planner import ChatCompletionsPlanner
    from .tools import market_tools

    st.title("Read-only Market Research")
    st.caption(
        "Experimental tool-assisted historical research; not financial advice. "
        "No account access or trades are available. Market prices are not live quotes."
    )
    question = st.text_area(
        "Question (include the ticker, such as RELIANCE.NS)",
        max_chars=2000,
        placeholder="What is the historical volatility of RELIANCE.NS over 6mo?",
    )
    if not st.button("Run research", type="primary"):
        return
    if not question.strip():
        st.warning("Enter a research question first.")
        return

    api_key = os.environ.get("ADVISOR_API_KEY", "")
    model = os.environ.get("ADVISOR_MODEL", "")
    endpoint = os.environ.get(
        "ADVISOR_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    if (not api_key or not model or api_key.startswith("replace-with-")
            or model.startswith("replace-with-")):
        st.error("Configure ADVISOR_API_KEY and ADVISOR_MODEL as environment variables.")
        return

    try:
        planner = ChatCompletionsPlanner(api_key, model, endpoint)
        with st.spinner("Gathering read-only market evidence..."):
            result = ResearchAgent(planner, market_tools(), max_steps=4).run(question)
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid research configuration: {exc}")
        return

    if result.status == "completed":
        st.markdown(result.answer)
    else:
        st.warning(result.answer)

    if result.observations:
        with st.expander("Historical evidence and tool status"):
            for observation in result.observations:
                st.json(
                    {
                        "tool": observation.tool,
                        "arguments": dict(observation.arguments),
                        "result": observation.result,
                        "error": observation.error,
                    }
                )
