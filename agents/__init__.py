"""
Agent package for the Multi-Agent Business Research Assistant.

Contains four specialised node functions, each responsible for a
distinct phase of the research pipeline:

    - **clarity_agent**:   Evaluates query clarity, triggers HITL interrupt
    - **research_agent**:  Tool-calling node with Tavily search
    - **validator_agent**: Critic node that evaluates research quality
    - **synthesis_agent**: Final report generation

Each agent function conforms to the LangGraph node signature::

    def agent_fn(state: AgentState) -> dict[str, Any]:
        ...
"""

from agents.clarity import clarity_agent
from agents.research import research_agent
from agents.validator import validator_agent
from agents.synthesis import synthesis_agent

__all__: list[str] = [
    "clarity_agent",
    "research_agent",
    "validator_agent",
    "synthesis_agent",
]
