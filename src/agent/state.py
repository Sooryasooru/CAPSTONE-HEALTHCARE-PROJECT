"""LangGraph state definition for the HAIP clinical agent.

The state is the agent's working memory — it carries the running list of
messages (user turns, LLM reasoning, tool calls, tool results) through every
node in the graph. LangGraph passes this object from node to node and merges
updates automatically via the `add_messages` reducer.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Working memory passed between graph nodes.

    Attributes:
        messages: The full conversation so far. `add_messages` is a reducer
            that appends new messages instead of overwriting, so each node can
            return just the message(s) it produced and LangGraph stitches the
            history together.
    """
    messages: Annotated[list, add_messages]