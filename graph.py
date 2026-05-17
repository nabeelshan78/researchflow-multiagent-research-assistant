"""
LangGraph StateGraph construction and conditional routing.

Builds the cyclical state machine that orchestrates the four agent
nodes with the following flow::

    START → clarity_agent ─┬─ "clear" ──────────→ research_agent
                           └─ "needs_clarification" → INTERRUPT → (resume) → clarity_agent
                                                          ↑ (re-evaluates after clarification)

    research_agent ─┬─ confidence >= threshold ──→ synthesis_agent → END
                    └─ confidence < threshold ───→ validator_agent
                                                       │
    validator_agent ─┬─ "sufficient" ────────────→ synthesis_agent → END
                     ├─ "insufficient" & attempts < max → research_agent (CYCLE)
                     └─ attempts >= max ─────────→ synthesis_agent → END

Graph Features:
    - MemorySaver checkpointing for interrupt/resume support
    - Conditional edges with named routing functions
    - Thread-based session management
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (
    clarity_agent,
    research_agent,
    synthesis_agent,
    validator_agent,
)
from config import settings
from state import AgentState

logger = logging.getLogger(__name__)


# ── Conditional routing functions ────────────────────────────────────────────

def route_after_clarity(state: AgentState) -> str:
    """Route based on the Clarity Agent's assessment.

    Args:
        state: Current graph state with ``clarity_status``.

    Returns:
        ``"research_agent"`` if the query is clear, or
        ``"clarity_agent"`` to re-evaluate (should not normally
        occur since interrupt handles the loop internally).
    """
    status = state.get("clarity_status", "")
    logger.debug("Routing after clarity: status=%s", status)

    if status == "clear":
        return "research_agent"

    # Fallback — should not reach here because interrupt() handles
    # the needs_clarification case within the clarity node itself.
    return "research_agent"


def route_after_research(state: AgentState) -> str:
    """Route based on the Research Agent's confidence score.

    Args:
        state: Current graph state with ``confidence_score``.

    Returns:
        ``"synthesis_agent"`` if confidence meets the threshold,
        ``"validator_agent"`` otherwise.
    """
    confidence = state.get("confidence_score", 0.0)
    threshold = settings.confidence_threshold

    logger.debug(
        "Routing after research: confidence=%.1f, threshold=%.1f",
        confidence,
        threshold,
    )

    if confidence >= threshold:
        logger.info(
            "Research confidence %.1f >= %.1f — skipping validation.",
            confidence,
            threshold,
        )
        return "synthesis_agent"

    return "validator_agent"


def route_after_validation(state: AgentState) -> str:
    """Route based on the Validator Agent's assessment and attempt count.

    Implements the cyclical loop:
    - ``"insufficient"`` AND attempts < max → back to research
    - ``"sufficient"`` OR attempts >= max → proceed to synthesis

    Args:
        state: Current graph state with ``validation_result``
            and ``research_attempts``.

    Returns:
        ``"research_agent"`` for another cycle, or
        ``"synthesis_agent"`` to finalise.
    """
    result = state.get("validation_result", "")
    attempts = state.get("research_attempts", 0)
    max_attempts = settings.max_research_attempts

    logger.debug(
        "Routing after validation: result=%s, attempts=%d/%d",
        result,
        attempts,
        max_attempts,
    )

    if result == "sufficient":
        return "synthesis_agent"

    if attempts >= max_attempts:
        logger.warning(
            "Max research attempts (%d) reached. Forcing synthesis.",
            max_attempts,
        )
        return "synthesis_agent"

    # Insufficient and under the cap — loop back for more research
    return "research_agent"


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the research assistant state graph.

    Creates a ``StateGraph`` with four nodes and three conditional
    edges implementing the research pipeline's cyclical logic.
    Uses ``MemorySaver`` for in-memory checkpointing to support
    the human-in-the-loop interrupt/resume workflow.

    Returns:
        A compiled ``StateGraph`` instance ready for invocation
        via ``.stream()`` or ``.invoke()``.

    Example::

        graph = build_graph()
        config = {"configurable": {"thread_id": "session-1"}}
        for event in graph.stream(initial_state, config):
            print(event)
    """
    logger.info("Building research assistant graph.")

    # ── Define the graph ─────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────
    builder.add_node("clarity_agent", clarity_agent)
    builder.add_node("research_agent", research_agent)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("synthesis_agent", synthesis_agent)

    # ── Entry edge ───────────────────────────────────────────────────
    builder.add_edge(START, "clarity_agent")

    # ── Conditional edges ────────────────────────────────────────────
    builder.add_conditional_edges(
        "clarity_agent",
        route_after_clarity,
        {
            "research_agent": "research_agent",
        },
    )

    builder.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "synthesis_agent": "synthesis_agent",
            "validator_agent": "validator_agent",
        },
    )

    builder.add_conditional_edges(
        "validator_agent",
        route_after_validation,
        {
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent",
        },
    )

    # ── Terminal edge ────────────────────────────────────────────────
    builder.add_edge("synthesis_agent", END)

    # ── Compile with checkpointer ────────────────────────────────────
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled successfully with MemorySaver checkpointer.")
    return graph
