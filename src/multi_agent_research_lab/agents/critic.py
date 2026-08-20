"""Critic agent implementation for fact-checking and citation validation."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking, citation validation, and hallucination inspection agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and calculate citation integrity."""
        if not state.final_answer:
            state.errors.append("Critic found no final_answer to critique.")
            return state

        # Compute citation indices found in final_answer e.g. [1], [2]
        cited_indices = set(re.findall(r"\[(\d+)\]", state.final_answer))
        num_sources = len(state.sources)
        valid_citations = [
            int(idx) for idx in cited_indices if idx.isdigit() and 1 <= int(idx) <= num_sources
        ]

        coverage = len(valid_citations) / num_sources if num_sources > 0 else 1.0

        system_prompt = (
            "You are a Quality & Fact-Checking Critic. Inspect the drafted report against "
            "the sources. Verify evidence grounding and citation coverage."
        )

        user_prompt = (
            f"Draft Report:\n{state.final_answer}\n\n"
            f"Number of Available Sources: {num_sources}\n"
            f"Detected Cited Sources: {valid_citations}\n\n"
            "Provide a brief quality rating (0-10) and notes on factual fidelity."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        result = AgentResult(
            agent=AgentName.CRITIC,
            content=response.content,
            metadata={
                "citation_coverage": coverage,
                "valid_citations_count": len(valid_citations),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "critic.done",
            {"citation_coverage": coverage, "cost_usd": response.cost_usd},
        )

        return state
