"""
Root graph — entry point registered in langgraph.json.

Composes Chat Agent (which internally embeds the Brain Agent).
The root graph's only job is to wire the entry message into OverallState
and expose the compiled graph as `graph`.

Input contract:
  Caller must supply at minimum:
    {"raw_user_input": "<user message>"}
  Optionally:
    {"loaded_resources": [...]}  if resources are already in session

Output:
  The graph updates OverallState in place.
  Callers read `state["final_answer"]` and `state["messages"]`.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import OverallState
from agent.brain_agent.graph import build_brain_agent_graph
from agent.chat_agent.graph import build_chat_agent_graph
from agent.rag import init_rag

# Pre-initialise Milvus, Neo4j and BGE-M3 before the async event loop starts.
# This avoids "blocking call" warnings during tool execution.
init_rag()


def reset_brain_state(state: OverallState) -> dict:
    """
    Reset Brain Agent fields at the start of every turn.

    Without this, fields like replan_count, plan, and executor_scratchpad
    carry over from the previous turn, causing the Brain Agent to behave
    incorrectly (e.g. replan_count already at max, so planning is skipped).
    """
    return {
        "plan": [],
        "current_step_index": 0,
        "replan_count": 0,
        "executor_scratchpad": [],
        "retrieved_context": [],
        "brain_done": False,
    }


def _build_root_graph(checkpointer=None, store=None):
    brain_graph = build_brain_agent_graph()
    chat_graph = build_chat_agent_graph(brain_agent_subgraph=brain_graph)

    builder = StateGraph(OverallState)
    builder.add_node("reset_brain_state", reset_brain_state)
    builder.add_node("chat_agent", chat_graph)

    builder.add_edge(START, "reset_brain_state")
    builder.add_edge("reset_brain_state", "chat_agent")
    builder.add_edge("chat_agent", END)

    return builder.compile(checkpointer=checkpointer, store=store)


# langgraph.json 指向这个变量（不带 checkpointer，用于 langgraph dev）
graph = _build_root_graph()