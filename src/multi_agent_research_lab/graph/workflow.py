"""LangGraph workflow implementation for the multi-agent research system."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and orchestrates the multi-agent research workflow using LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(settings=self.settings)
        self.search_client = search_client or SearchClient(settings=self.settings)

        self.supervisor = SupervisorAgent(settings=self.settings)
        self.researcher = ResearcherAgent(
            search_client=self.search_client, llm_client=self.llm_client
        )
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)

        self._compiled_graph: Any = None

    def build(self) -> Any:
        """Create and compile the LangGraph workflow graph."""
        builder: StateGraph = StateGraph(ResearchState)

        # 1. Register agent nodes
        builder.add_node("supervisor", self.supervisor.run)
        builder.add_node("researcher", self.researcher.run)
        builder.add_node("analyst", self.analyst.run)
        builder.add_node("writer", self.writer.run)
        builder.add_node("critic", self.critic.run)

        # 2. Add edges: START -> supervisor
        builder.add_edge(START, "supervisor")

        # 3. Add conditional router from supervisor
        def router(state: ResearchState) -> str:
            return self.supervisor.route(state)

        builder.add_conditional_edges(
            "supervisor",
            router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # 4. Return control back to supervisor after each worker execution
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the LangGraph workflow and return the final ResearchState."""
        if self._compiled_graph is None:
            self.build()

        result = self._compiled_graph.invoke(state)

        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        else:
            return ResearchState(
                request=state.request,
                iteration=getattr(result, "iteration", state.iteration),
                route_history=getattr(result, "route_history", state.route_history),
                sources=getattr(result, "sources", state.sources),
                research_notes=getattr(result, "research_notes", state.research_notes),
                analysis_notes=getattr(result, "analysis_notes", state.analysis_notes),
                final_answer=getattr(result, "final_answer", state.final_answer),
                agent_results=getattr(result, "agent_results", state.agent_results),
                trace=getattr(result, "trace", state.trace),
                errors=getattr(result, "errors", state.errors),
            )
