"""Unit tests for agents and routing policy."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routing_sequence() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()

    # 1. When no sources -> route to researcher
    assert supervisor.route(state) == "researcher"

    # 2. When sources exist but no analysis -> route to analyst
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    assert supervisor.route(state) == "analyst"

    # 3. When analysis exists but no final_answer -> route to writer
    state.analysis_notes = "Key analysis notes"
    assert supervisor.route(state) == "writer"

    # 4. When final_answer exists -> route to done
    state.final_answer = "Final answer [1]"
    assert supervisor.route(state) == "done"


def test_supervisor_max_iterations_guardrail() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 10  # exceeds max_iterations (default 6)
    supervisor = SupervisorAgent()
    assert supervisor.route(state) == "done"


def test_researcher_agent_fallback() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    researcher = ResearcherAgent()
    updated_state = researcher.run(state)
    assert len(updated_state.sources) > 0
    assert updated_state.research_notes is not None


def test_analyst_agent() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.research_notes = "Finding 1: Multi-agent reduces context fragmentation."
    analyst = AnalystAgent()
    updated_state = analyst.run(state)
    assert updated_state.analysis_notes is not None


def test_writer_agent() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [
        SourceDocument(title="Source A", url="https://example.com/a", snippet="Overview")
    ]
    state.analysis_notes = "Key point: graphrag improves global reasoning."
    writer = WriterAgent()
    updated_state = writer.run(state)
    assert updated_state.final_answer is not None


def test_critic_agent() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [
        SourceDocument(title="Source A", url="https://example.com/a", snippet="Overview")
    ]
    state.final_answer = "According to research [1], GraphRAG outperforms standard RAG."
    critic = CriticAgent()
    updated_state = critic.run(state)
    assert len(updated_state.agent_results) > 0


def test_multi_agent_workflow_build() -> None:
    workflow = MultiAgentWorkflow()
    graph = workflow.build()
    assert graph is not None
