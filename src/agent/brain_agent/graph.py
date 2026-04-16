"""
Brain Agent subgraph — Plan-and-Execute + Reflection.

Uses OverallState directly (方案 A shared state).

Node flow:
  START
    -> plan           (Planner LLM: produce or revise the plan)
    -> execute_step   (Executor: call the tool for the current step)
    -> evaluate_step  (Executor: judge success or failure)
         |-- success, more steps --> advance_step --> execute_step
         |-- success, last step  --> assemble_context
         |-- failed, retries left --> execute_step   (retry same step)
         |-- failed, replan budget --> plan          (escalate)
         |-- failed, exhausted    --> assemble_context
    -> assemble_context
    -> END

Brain Agent only writes these fields:
  plan, current_step_index, replan_count, executor_scratchpad,
  retrieved_context, brain_done
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from agent.state import OverallState, ExecutionStep, RetrievedChunk
from agent.configuration import Configuration
from agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REPLAN_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    EXECUTOR_STEP_PROMPT,
    format_resources,
)
from agent.tools_and_schemas import (
    ExecutionPlan,
    RefinedToolCall,
    TOOL_MAP,
    get_tool_descriptions,
)
from agent.utils import get_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_scratchpad(scratchpad: list[dict]) -> str:
    """
    Format executor_scratchpad into a readable string for the Executor LLM.

    Truncation strategy: keep the full content of the most recent 3 entries,
    summarise older entries as one-liners. This ensures the immediately prior
    step results (most likely to be needed) are never truncated, while keeping
    total length manageable.
    """
    if not scratchpad:
        return "(No prior steps executed yet)"

    lines = []
    recent = scratchpad[-3:]
    older = scratchpad[:-3]

    # Older entries: one-line summary only
    for entry in older:
        if entry["type"] == "success":
            lines.append(f"Step {entry['step_id']} SUCCESS (result omitted, too old)")
        else:
            lines.append(f"Step {entry['step_id']} ERROR: {entry.get('error_report', '')}")

    # Recent entries: full result
    for entry in recent:
        if entry["type"] == "success":
            lines.append(f"Step {entry['step_id']} SUCCESS:\n{entry.get('result', '')}")
        else:
            lines.append(f"Step {entry['step_id']} ERROR: {entry.get('error_report', '')}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def plan(state: OverallState, config: RunnableConfig) -> dict:
    """
    Planner node: produce an initial plan, or revise after an error.
    First call (replan_count == 0): generate fresh plan.
    Subsequent calls: revise based on the error in executor_scratchpad.
    """
    cfg = Configuration.from_runnable_config(config)
    llm = get_llm(cfg.brain_model)
    structured_llm = llm.with_structured_output(ExecutionPlan)

    system_msg = SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(
        tool_descriptions=get_tool_descriptions(),
        resources=format_resources(state.loaded_resources),
    ))

    if state.replan_count == 0:
        status = "正在规划执行步骤..."
        user_msg = HumanMessage(content=PLANNER_USER_PROMPT.format(
            query=state.normalized_query,
        ))
    else:
        status = "执行遇到问题，正在调整计划..."
        last_error = next(
            (s for s in reversed(state.executor_scratchpad) if s.get("type") == "error"),
            {"error_report": "Unknown error"},
        )
        user_msg = HumanMessage(content=REPLAN_PROMPT.format(
            query=state.normalized_query,
            original_plan=json.dumps([s.model_dump() for s in state.plan], indent=2),
            error_report=last_error.get("error_report", ""),
        ))

    result: ExecutionPlan = await structured_llm.ainvoke([system_msg, user_msg])

    if not result.steps:
        return {
            "plan": [],
            "current_step_index": 0,
            "replan_count": state.replan_count + 1,
            "executor_scratchpad": [],
            "status_message": status,
        }

    new_plan = [
        ExecutionStep(
            step_id=s.step_id,
            description=s.description,
            tool_name=s.tool_name,
            tool_args=s.tool_args,
            status="pending",
        )
        for s in result.steps
    ]

    # Note: last_error is read from executor_scratchpad BEFORE this return,
    # so clearing it here is safe — the error info is already in user_msg.
    return {
        "plan": new_plan,
        "current_step_index": 0,
        "replan_count": state.replan_count + 1,
        "executor_scratchpad": [],
        "status_message": status,
    }


async def execute_step(state: OverallState, config: RunnableConfig) -> dict:
    """
    Executor node: use an LLM to refine tool_args based on prior step results,
    then call the tool.

    Flow:
      1. Executor LLM sees current step + scratchpad → outputs RefinedToolCall
         (may adjust tool_args if prior results are needed as input)
      2. Resolved tool function is called with the refined args
      3. Success/failure recorded into plan and scratchpad
    """
    cfg = Configuration.from_runnable_config(config)
    step = state.plan[state.current_step_index]
    updated_plan = list(state.plan)
    status = f"正在执行：{step.description}"

    # Mark running
    updated_plan[state.current_step_index] = step.model_copy(update={"status": "running"})

    # -- Tool existence check (fast fail before calling LLM) ---------------
    tool_fn = TOOL_MAP.get(step.tool_name or "")
    if tool_fn is None:
        err = f"Tool '{step.tool_name}' not found in TOOL_MAP."
        updated_plan[state.current_step_index] = step.model_copy(
            update={"status": "failed", "error_msg": err}
        )
        return {
            "plan": updated_plan,
            "status_message": status,
            "executor_scratchpad": state.executor_scratchpad + [
                {"type": "error", "step_id": step.step_id, "error_report": err}
            ],
        }

    # -- Executor LLM: refine tool_args using prior scratchpad results ------
    try:
        llm = get_llm(cfg.brain_model)
        structured_llm = llm.with_structured_output(RefinedToolCall)

        scratchpad_str = _format_scratchpad(state.executor_scratchpad)

        system_msg = SystemMessage(content=EXECUTOR_SYSTEM_PROMPT.format(
            tool_descriptions=get_tool_descriptions(),
        ))
        user_msg = HumanMessage(content=EXECUTOR_STEP_PROMPT.format(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            scratchpad=scratchpad_str,
        ))

        refined: RefinedToolCall = await structured_llm.ainvoke([system_msg, user_msg])
        final_args = refined.tool_args

    except Exception as exc:
        # If the Executor LLM itself fails, fall back to Planner's original args
        final_args = step.tool_args

    # -- Actually call the tool with the (possibly refined) args ------------
    try:
        result = await tool_fn.ainvoke(final_args)
        updated_plan[state.current_step_index] = step.model_copy(
            update={"status": "success", "result": result, "tool_args": final_args}
        )
        scratchpad_entry = {"type": "success", "step_id": step.step_id, "result": result}
    except Exception as exc:
        err = str(exc)
        updated_plan[state.current_step_index] = step.model_copy(
            update={"status": "failed", "error_msg": err, "tool_args": final_args}
        )
        scratchpad_entry = {"type": "error", "step_id": step.step_id, "error_report": err}

    return {
        "plan": updated_plan,
        "status_message": status,
        "executor_scratchpad": state.executor_scratchpad + [scratchpad_entry],
    }


async def evaluate_step(state: OverallState) -> dict:
    """
    Evaluation node: currently rule-based (success/fail from step.status).
    Increments retry_count if the step failed, so the router can decide
    whether to retry or escalate to the planner.
    """
    step = state.plan[state.current_step_index]
    if step.status == "failed":
        updated_plan = list(state.plan)
        updated_plan[state.current_step_index] = step.model_copy(
            update={"retry_count": step.retry_count + 1}
        )
        return {"plan": updated_plan}
    return {}


def advance_step(state: OverallState) -> dict:
    """Increment the step pointer. Called when moving to the next step."""
    return {"current_step_index": state.current_step_index + 1}


async def assemble_context(state: OverallState) -> dict:
    """
    Final Brain Agent node: collect RetrievedChunk objects from all successful
    tool calls and write them to retrieved_context.

    Accepts any tool result that is a list of dicts containing 'node_id' and
    'resource_type' — covers rag_retrieve, search_github_repo, and future tools.
    """
    CHUNK_RESOURCE_TYPES = {"video_slide", "paper_chunk", "github_repo", "search_result", "job_result"}

    all_chunks: list[RetrievedChunk] = []
    job_messages: list[str] = []

    for step in state.plan:
        if step.status != "success":
            continue
        # Background job result (download_and_process_*)
        if isinstance(step.result, dict) and "job_id" in step.result:
            job_messages.append(step.result.get("message", f"任务已提交，job_id: {step.result['job_id']}"))
            continue
        if not isinstance(step.result, list):
            continue
        for item in step.result:
            if not isinstance(item, dict):
                continue
            if item.get("resource_type") not in CHUNK_RESOURCE_TYPES:
                continue
            try:
                all_chunks.append(RetrievedChunk(**item))
            except Exception:
                pass

    # Pass job submission info to render_response via a synthetic chunk
    if job_messages:
        all_chunks.append(RetrievedChunk(
            node_id="job_submission",
            resource_id="system",
            resource_type="job_result",
            content="\n".join(job_messages),
            summary="",
            score=1.0,
            location="",
        ))

    return {
        "retrieved_context": all_chunks,
        "brain_done": True,
        "status_message": "正在整理检索结果...",
    }


def route_after_plan(
    state: OverallState,
) -> Literal["execute_step", "assemble_context"]:
    """Route after plan node: skip execution if plan is empty."""
    if not state.plan:
        return "assemble_context"
    return "execute_step"

def route_after_evaluate(
    state: OverallState,
    config: RunnableConfig,
) -> Literal["advance_step", "execute_step", "plan", "assemble_context"]:
    """
    Decision logic after step evaluation:
      success + more steps  -> advance_step (then execute_step)
      success + last step   -> assemble_context
      failed + retries left -> execute_step  (retry, no advance)
      failed + replan left  -> plan
      failed + exhausted    -> assemble_context
    """
    cfg = Configuration.from_runnable_config(config)
    step = state.plan[state.current_step_index]

    if step.status == "success":
        if state.current_step_index + 1 < len(state.plan):
            return "advance_step"
        return "assemble_context"

    # Failed
    if step.retry_count < cfg.max_executor_retries:
        return "execute_step"
    if state.replan_count < cfg.max_replan:
        return "plan"
    return "assemble_context"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_brain_agent_graph() -> StateGraph:
    """Build and compile the Brain Agent subgraph."""
    builder = StateGraph(OverallState)

    builder.add_node("plan", plan)
    builder.add_node("execute_step", execute_step)
    builder.add_node("evaluate_step", evaluate_step)
    builder.add_node("advance_step", advance_step)
    builder.add_node("assemble_context", assemble_context)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "execute_step": "execute_step",
            "assemble_context": "assemble_context",
        },
    )
    builder.add_edge("execute_step", "evaluate_step")

    builder.add_conditional_edges(
        "evaluate_step",
        route_after_evaluate,
        {
            "advance_step": "advance_step",
            "execute_step": "execute_step",
            "plan": "plan",
            "assemble_context": "assemble_context",
        },
    )
    builder.add_edge("advance_step", "execute_step")
    builder.add_edge("assemble_context", END)

    return builder.compile()