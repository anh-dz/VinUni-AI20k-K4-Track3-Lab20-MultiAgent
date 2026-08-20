"""Supervisor / router implementation."""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def route(self, state: ResearchState) -> str:
        """Determine next node in workflow: researcher, analyst, writer, or done."""
        # Guardrail: stop when max iterations reached to prevent infinite loops
        if state.iteration >= self.settings.max_iterations:
            return "done"

        # Sequential handoff logic based on state fields
        if not state.sources:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route and record trace."""
        next_route = self.route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.route",
            {
                "iteration": state.iteration,
                "next_route": next_route,
                "sources_count": len(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
