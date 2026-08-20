"""Analyst agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights and comparative analysis."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` by critically evaluating research findings."""
        if not state.research_notes and not state.sources:
            state.errors.append("Analyst received empty research notes and sources.")
            state.analysis_notes = "Insufficient research information available to analyze."
            return state

        system_prompt = (
            "You are a Senior Technical Research Analyst. Critically assess raw research "
            "findings. Identify key claims, compare different approaches, evaluate trade-offs, "
            "highlight limitations, and organize information into structured insights."
        )

        user_prompt = (
            f"Original Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Raw Research Notes:\n{state.research_notes or 'None'}\n\n"
            "Please provide a rigorous structured analysis covering:\n"
            "1. Key Technical Insights & Architecture\n"
            "2. Trade-offs & Comparisons (Latency, Cost, Accuracy, Scalability)\n"
            "3. Failure Modes & Limitations\n"
            "4. Practical Recommendations"
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content

        result = AgentResult(
            agent=AgentName.ANALYST,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "analyst.done",
            {"cost_usd": response.cost_usd},
        )

        return state
