"""
Chat Agent subgraph.

Uses OverallState directly (方案 A shared state).

Flow:
  START -> load_memory -> normalize_input -> brain_agent -> render_response -> save_memory -> END

Chat Agent owns:
  raw_user_input (read), normalized_query (write),
  final_answer (write), messages (write), long_term_memory (write via load_memory)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import (
    CHAT_NORMALIZE_PROMPT,
    CHAT_RENDER_PROMPT,
    format_retrieved_context,
    is_note_request,
)
from agent.utils import get_llm
from agent.memory import load_memory, save_memory
from agent.eval_logger import log_eval_sample


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def normalize_input(state: OverallState, config: RunnableConfig) -> dict:
    """
    Rewrite raw_user_input into a clean, self-contained normalized_query.
    Resolves pronouns and references using the last few messages as context.
    Also injects long_term_memory (loaded by the preceding load_memory node).
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.chat_model)

    # Exclude the last message (current turn's HumanMessage already in state.messages)
    # to avoid duplicating raw_user_input in the history string.
    # Contract: caller must append the current HumanMessage to messages before invoking,
    # so messages[-1] is always the current turn's user message.
    history_turns = state.messages[:-1][-6:]
    history_str = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history_turns
    ) or "(start of conversation)"

    prompt = CHAT_NORMALIZE_PROMPT.format(
        memory=state.long_term_memory or "(no memories yet)",
        history=history_str,
        raw_input=state.raw_user_input,
    )

    response = await llm.ainvoke(prompt)
    return {
        "normalized_query": response.content.strip(),
        "status_message": "正在理解您的问题...",
    }


async def render_response(state: OverallState, config: RunnableConfig) -> dict:
    """
    Turn the retrieved context assembled by the Brain Agent into a
    fluent natural-language answer, then append it to messages.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.chat_model)

    prompt = CHAT_RENDER_PROMPT.format(
        query=state.normalized_query,
        context=format_retrieved_context(state.retrieved_context),
    )

    response = await llm.ainvoke(prompt)
    answer = response.content.strip()

    return {
        "final_answer": answer,
        "status_message": "正在生成回答...",
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def route_after_normalize(state: OverallState) -> Literal["note_agent", "brain_agent"]:
    """Route to Note Agent if the user asked for study notes, else Brain Agent."""
    if is_note_request(state.raw_user_input):
        return "note_agent"
    return "brain_agent"


def build_chat_agent_graph(brain_agent_subgraph) -> StateGraph:
    """
    Build and compile the Chat Agent subgraph.

    Args:
        brain_agent_subgraph: compiled Brain Agent graph, embedded as a node.
            Because both graphs use OverallState, LangGraph passes the full
            state through with zero field mapping required.
    """
    from agent.note_agent.graph import build_note_agent_graph
    note_agent_subgraph = build_note_agent_graph()

    builder = StateGraph(OverallState)

    builder.add_node("load_memory", load_memory)
    builder.add_node("normalize_input", normalize_input)
    builder.add_node("brain_agent", brain_agent_subgraph)
    builder.add_node("note_agent", note_agent_subgraph)
    builder.add_node("render_response", render_response)
    builder.add_node("save_memory", save_memory)
    builder.add_node("log_eval_sample", log_eval_sample)

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "normalize_input")
    builder.add_conditional_edges(
        "normalize_input",
        route_after_normalize,
        {"note_agent": "note_agent", "brain_agent": "brain_agent"},
    )
    # note_agent bypasses render_response (it writes final_answer directly)
    builder.add_edge("note_agent", "save_memory")
    builder.add_edge("brain_agent", "render_response")
    builder.add_edge("render_response", "save_memory")
    builder.add_edge("save_memory", "log_eval_sample")
    builder.add_edge("log_eval_sample", END)

    return builder.compile()
