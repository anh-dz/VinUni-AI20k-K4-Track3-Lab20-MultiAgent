"""Writer agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces the final comprehensive answer with rigorous source citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` by synthesizing analysis and research notes."""
        context = state.analysis_notes or state.research_notes or "No notes provided."

        sources_ref = "\n".join(
            f"[{i + 1}] {doc.title} - URL: {doc.url or 'N/A'}\nSummary: {doc.snippet}"
            for i, doc in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Principal Technical Writer. Craft a polished, authoritative research "
            "report based on provided analysis and sources. "
            "CRITICAL: Every major factual claim must be cited inline like [1], [2]. "
            "At the bottom, include a '## References' section listing each cited source."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analysis Notes:\n{context}\n\n"
            f"Available Sources for Citation:\n{sources_ref}\n\n"
            "Please write the final report with clear headings, inline citations [1], [2], "
            "and a References section."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = response.content

        result = AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "writer.done",
            {"cost_usd": response.cost_usd},
        )

        return state
