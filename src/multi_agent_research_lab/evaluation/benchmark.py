"""Benchmark evaluation for single-agent vs multi-agent research systems."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate ratio of available sources cited in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    cited_count = 0
    answer_lower = state.final_answer.lower()
    bracket_numbers = set(re.findall(r"\[(\d+)\]", state.final_answer))

    for i, doc in enumerate(state.sources):
        index_str = str(i + 1)
        if index_str in bracket_numbers or (doc.title and doc.title.lower() in answer_lower):
            cited_count += 1

    return min(1.0, cited_count / len(state.sources))


def compute_total_cost(state: ResearchState) -> float:
    """Sum estimated USD cost across all agent results."""
    total = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost and isinstance(cost, (int, float)):
            total += cost
    return total


def compute_quality_score(state: ResearchState) -> float:
    """Heuristic quality scoring on 0-10 scale based on structure, depth, citations and errors."""
    if not state.final_answer:
        return 0.0

    score = 4.0  # baseline for producing an answer

    # Length & depth check
    word_count = len(state.final_answer.split())
    if word_count >= 250:
        score += 2.0
    elif word_count >= 100:
        score += 1.0

    # Markdown structure (headings, bullet points, sections)
    if "##" in state.final_answer:
        score += 1.5
    if "- " in state.final_answer or "* " in state.final_answer or "1." in state.final_answer:
        score += 0.5

    # Citations & references section
    cov = compute_citation_coverage(state)
    score += cov * 2.0

    # Penalize errors
    if state.errors:
        score = max(0.0, score - 1.5 * len(state.errors))

    return min(10.0, round(score, 1))


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner and calculate latency, cost, quality, and citation metrics."""
    started = perf_counter()
    failure_rate = 0.0
    error_note = ""

    try:
        state = runner(query)
        if not state.final_answer or len(state.final_answer.strip()) == 0:
            failure_rate = 1.0
            error_note = "Empty final answer"
    except Exception as exc:
        failure_rate = 1.0
        error_note = str(exc)
        state = ResearchState(
            request=getattr(state if "state" in locals() else None, "request", None),
            errors=[str(exc)],
        )

    latency = perf_counter() - started
    cost = compute_total_cost(state)
    cov = compute_citation_coverage(state)
    quality = compute_quality_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=cov,
        failure_rate=failure_rate,
        notes=error_note or f"Routes: {len(state.route_history)} steps",
    )

    return state, metrics
