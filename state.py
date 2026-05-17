"""
LangGraph state schema for the Multi-Agent Business Research Assistant.

Defines the central ``AgentState`` TypedDict that flows through every
node in the graph.  LangGraph's ``Annotated`` + ``add_messages`` reducer
is used for the ``messages`` field so that each node can *append*
messages without overwriting the full list.

State Fields at a Glance::

    messages              — Conversation history (auto-merged via reducer)
    user_query            — The current user research query
    clarity_status        — "clear" | "needs_clarification" | ""
    clarification_question— Question to ask the user (set by Clarity Agent)
    research_data         — Accumulated search findings
    confidence_score      — Research self-evaluation score (0-10)
    research_attempts     — Counter for research → validate cycles
    validation_result     — "sufficient" | "insufficient" | ""
    validation_reasoning  — Validator's reasoning for its decision
    final_report          — Formatted synthesis output
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchFinding(TypedDict):
    """A single research finding from a Tavily search result.

    Attributes:
        source: The URL or name of the source.
        content: The extracted text content from the source.
        query: The search query that produced this finding.
    """

    source: str
    content: str
    query: str


class AgentState(TypedDict):
    """Central state schema for the research assistant graph.

    This TypedDict is the single source of truth that flows through
    every node.  Each agent reads the fields it needs and writes back
    its outputs.  LangGraph handles merging via the ``add_messages``
    reducer on the ``messages`` field.

    Routing Fields:
        ``clarity_status``, ``confidence_score``, ``validation_result``,
        and ``research_attempts`` are used by conditional edge functions
        to determine the next node in the graph.

    Lifecycle:
        1. **Clarity Agent** → sets ``clarity_status``, optionally
           ``clarification_question``
        2. **Research Agent** → appends to ``research_data``, sets
           ``confidence_score``
        3. **Validator Agent** → sets ``validation_result``, increments
           ``research_attempts``
        4. **Synthesis Agent** → sets ``final_report``

    Attributes:
        messages: Conversation history managed by LangGraph's
            ``add_messages`` reducer.  Nodes append ``AIMessage`` /
            ``HumanMessage`` instances; the reducer merges them
            automatically.
        user_query: The raw research question from the user.
        clarity_status: Routing flag set by the Clarity Agent.
            One of ``"clear"``, ``"needs_clarification"``, or ``""``
            (uninitialised).
        clarification_question: If ``clarity_status`` is
            ``"needs_clarification"``, this contains the question the
            system wants to ask the user.
        research_data: Accumulated list of :class:`ResearchFinding`
            dicts.  The Research Agent appends to this on each cycle.
        confidence_score: Self-evaluated confidence (0.0–10.0) from the
            Research Agent.  Scores ≥ ``settings.confidence_threshold``
            bypass validation.
        research_attempts: How many research → validate cycles have
            occurred.  Capped at ``settings.max_research_attempts``.
        validation_result: Routing flag set by the Validator Agent.
            One of ``"sufficient"``, ``"insufficient"``, or ``""``
            (uninitialised).
        validation_reasoning: The Validator's textual justification
            for its decision, useful for debugging and tracing.
        final_report: The formatted research report produced by the
            Synthesis Agent.  This is the primary output presented
            to the user.
    """

    # ── Conversation Memory ──────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Query ────────────────────────────────────────────────────────────
    user_query: str

    # ── Clarity Routing ──────────────────────────────────────────────────
    clarity_status: str
    clarification_question: str

    # ── Research Payload ─────────────────────────────────────────────────
    research_data: list[ResearchFinding]
    confidence_score: float

    # ── Validation Routing ───────────────────────────────────────────────
    research_attempts: int
    validation_result: str
    validation_reasoning: str

    # ── Final Output ─────────────────────────────────────────────────────
    final_report: str
