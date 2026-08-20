"""Researcher agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # 1. Search for relevant sources
        sources = self.search_client.search(query=query, max_results=max_sources)
        state.sources = sources

        if not sources:
            state.errors.append("Researcher found no sources for query.")
            state.research_notes = "No relevant sources could be retrieved."
            return state

        # 2. Prepare context for LLM summarization
        sources_text = "\n\n".join(
            f"Source [{i + 1}] Title: {doc.title}\nURL: {doc.url or 'N/A'}\nSnippet: {doc.snippet}"
            for i, doc in enumerate(sources)
        )

        system_prompt = (
            "You are an expert research investigator. Your role is to examine the provided "
            "sources and extract key facts, methodologies, and core data into concise research "
            "notes. Do not hallucinate; ground your notes strictly in the given sources."
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Retrieved Sources:\n{sources_text}\n\n"
            "Please compile structured research notes summarizing findings from these sources."
        )

        # 3. Call LLM to synthesize research notes
        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = response.content

        # 4. Record AgentResult and trace event
        result = AgentResult(
            agent=AgentName.RESEARCHER,
            content=response.content,
            metadata={
                "num_sources": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        state.agent_results.append(result)
        state.add_trace_event(
            "researcher.done",
            {"num_sources": len(sources), "cost_usd": response.cost_usd},
        )

        return state
