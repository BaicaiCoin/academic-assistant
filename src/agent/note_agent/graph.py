"""
Note Agent subgraph.

Triggered when the user asks to generate study notes (e.g. "帮我生成笔记",
"总结一下今天的学习").

Flow:
  START -> generate_notes -> END

Reads:
  state.messages        — full conversation history for this thread
  state.raw_user_input  — to detect the note request

Writes:
  state.final_answer    — the generated Markdown notes
  state.messages        — appends AIMessage with the notes
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import NOTE_GENERATION_PROMPT
from agent.utils import get_llm


async def generate_notes(state: OverallState, config: RunnableConfig) -> dict:
    """
    Summarise the conversation history into structured Markdown study notes.
    Skips turns that are not Q&A (e.g. download requests, greetings).
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.chat_model)

    # Build a clean Q&A transcript from the message history
    pairs: list[str] = []
    messages = state.messages
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            # Find the next AI message as the answer
            answer = ""
            for j in range(i + 1, len(messages)):
                if isinstance(messages[j], AIMessage):
                    answer = messages[j].content
                    break
            if answer:
                pairs.append(f"Q: {msg.content}\nA: {answer}")

    if not pairs:
        note = "（当前会话暂无可用的问答记录，无法生成笔记。）"
        return {
            "final_answer": note,
            "status_message": "笔记生成完成",
            "messages": [AIMessage(content=note)],
        }

    transcript = "\n\n---\n\n".join(pairs)

    prompt = NOTE_GENERATION_PROMPT.format(transcript=transcript)
    response = await llm.ainvoke(prompt)
    note = response.content.strip()

    return {
        "final_answer": note,
        "status_message": "笔记生成完成",
        "messages": [AIMessage(content=note)],
    }


def build_note_agent_graph() -> StateGraph:
    builder = StateGraph(OverallState)
    builder.add_node("generate_notes", generate_notes)
    builder.add_edge(START, "generate_notes")
    builder.add_edge("generate_notes", END)
    return builder.compile()
