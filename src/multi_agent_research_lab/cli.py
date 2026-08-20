"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    import os

    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint


def _parse_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchQuery:
    try:
        return ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent_baseline(query_text: str) -> ResearchState:
    """Execute single-agent baseline: 1 direct LLM call without external search."""
    request = ResearchQuery(query=query_text)
    state = ResearchState(request=request)
    llm = LLMClient()

    system_prompt = (
        "You are an AI research assistant. Answer the user query directly and comprehensively."
    )
    resp = llm.complete(system_prompt=system_prompt, user_prompt=query_text)
    state.final_answer = resp.content
    state.record_route("single_agent_llm")
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=resp.content,
            metadata={
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
            },
        )
    )
    return state


def run_multi_agent_system(query_text: str) -> ResearchState:
    """Execute full multi-agent workflow."""
    request = ResearchQuery(query=query_text)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run single-agent baseline (single direct LLM call)."""
    _init()
    console.print(f"[bold cyan]Running Single-Agent Baseline:[/bold cyan] {query}\n")

    started = perf_counter()
    state = run_single_agent_baseline(query)
    elapsed = perf_counter() - started

    cost = sum(r.metadata.get("cost_usd", 0.0) for r in state.agent_results)

    console.print(Panel(Markdown(state.final_answer or ""), title="Single-Agent Baseline Answer"))
    console.print(
        f"[green]✔ Done in {elapsed:.2f}s | Cost: ${cost:.6f} | "
        f"Steps: {len(state.route_history)}[/green]"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", "-m", help="Max search sources")] = 5,
) -> None:
    """Run full Multi-Agent Research System (Supervisor -> Researcher -> Analyst -> Writer)."""
    _init()
    console.print(f"[bold green]Running Multi-Agent Research System:[/bold green] {query}\n")

    req = _parse_query(query, max_sources=max_sources)
    state = ResearchState(request=req)
    workflow = MultiAgentWorkflow()

    try:
        started = perf_counter()
        final_state = workflow.run(state)
        elapsed = perf_counter() - started
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Display final report
    console.print(
        Panel(Markdown(final_state.final_answer or ""), title="Multi-Agent Research Report")
    )

    # Display execution metadata
    table = Table(title="Workflow Execution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    total_cost = sum(r.metadata.get("cost_usd", 0.0) for r in final_state.agent_results)
    table.add_row("Total Latency", f"{elapsed:.2f}s")
    table.add_row("Total Estimated Cost", f"${total_cost:.6f}")
    table.add_row("Sources Retrieved", str(len(final_state.sources)))
    table.add_row("Routing Steps", " -> ".join(final_state.route_history))
    table.add_row("Iterations", str(final_state.iteration))

    console.print(table)


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query to benchmark"),
    ] = "Research GraphRAG state-of-the-art and write a summary",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Path to save benchmark report markdown"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run benchmark comparing Single-Agent Baseline vs Multi-Agent Workflow."""
    _init()
    console.print(f"[bold blue]Running Benchmark Suite:[/bold blue] '{query}'\n")

    metrics_list: list[BenchmarkMetrics] = []

    # 1. Benchmark Single-Agent
    console.print("[yellow]Benchmarking Single-Agent Baseline...[/yellow]")
    _, m_single = run_benchmark("Single-Agent Baseline", query, run_single_agent_baseline)
    metrics_list.append(m_single)
    console.print(
        f"  Single-Agent Latency: {m_single.latency_seconds:.2f}s, "
        f"Quality: {m_single.quality_score}/10"
    )

    # 2. Benchmark Multi-Agent
    console.print("[yellow]Benchmarking Multi-Agent System...[/yellow]")
    _, m_multi = run_benchmark("Multi-Agent Research System", query, run_multi_agent_system)
    metrics_list.append(m_multi)
    console.print(
        f"  Multi-Agent Latency: {m_multi.latency_seconds:.2f}s, "
        f"Quality: {m_multi.quality_score}/10, "
        f"Citation Coverage: {m_multi.citation_coverage:.0%}"
    )

    # Render report
    md_report = render_markdown_report(metrics_list)
    console.print("\n" + md_report)

    # Save to file
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_report, encoding="utf-8")
    console.print(f"[bold green]✔ Benchmark report saved to:[/bold green] {out_path}")


if __name__ == "__main__":
    app()
