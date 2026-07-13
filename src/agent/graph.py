r"""LangGraph clinical agent for HAIP.

This builds a ReAct-style agent: the LLM reasons about the question, decides
whether to call a tool, we run the tool, feed the result back, and let the LLM
reason again — looping until it has enough to answer. The loop is bounded by a
fixed tool whitelist and a safety-first system prompt, matching the Week 1
"deterministic router with whitelisted tools" design.

Graph shape:

    START -> agent -> (tools?) -> agent -> ... -> END
                 \-> (no tool) -> END
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

load_dotenv()
# Your .env uses GEMINI_API_KEY, but langchain-google-genai looks for
# GOOGLE_API_KEY. We read the former and pass it in explicitly, so neither
# the .env nor the rest of the codebase has to change.
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model, bound to the tool whitelist.

    `bind_tools` tells Gemini which tools exist and their schemas, so it can
    emit structured tool calls. It can ONLY call these — nothing else.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,  # low: clinical work wants consistency, not creativity
        google_api_key=GEMINI_KEY,
    )
    return llm.bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState) -> dict:
    """The reasoning node: prepend the system prompt and ask the LLM."""
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    """Route: if the LLM asked for a tool, run it; otherwise we're done."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_agent():
    """Compile and return the runnable LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after a tool runs, reason again

    compiled = graph.compile()
    logger.info("HAIP agent graph compiled with %d tools", len(ALL_TOOLS))
    return compiled


# Singleton — build once, reuse across requests.
AGENT = build_agent()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from langchain_core.messages import HumanMessage

    q = "What is the sepsis screening protocol, and what are our current KPIs?"
    result = AGENT.invoke({"messages": [HumanMessage(content=q)]})
    print("\n" + "=" * 70)
    print("FINAL ANSWER:\n", result["messages"][-1].content)