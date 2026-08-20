"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with online Tavily and offline corpus fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._tavily_client: object | None = None

        if self.settings.tavily_api_key:
            try:
                from tavily import TavilyClient

                self._tavily_client = TavilyClient(api_key=self.settings.tavily_api_key)
            except Exception as exc:
                logger.warning(f"Could not initialize Tavily client: {exc}")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily_client is not None:
            try:
                return self._search_tavily(query, max_results=max_results)
            except Exception as exc:
                logger.warning(f"Tavily search failed ({exc}). Falling back to offline corpus.")

        return self._search_offline_corpus(query, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        from tavily import TavilyClient

        client: TavilyClient = self._tavily_client  # type: ignore[assignment]
        response = client.search(query=query, max_results=max_results)
        results = response.get("results", [])

        documents: list[SourceDocument] = []
        for r in results[:max_results]:
            doc = SourceDocument(
                title=r.get("title") or "Untitled Source",
                url=r.get("url"),
                snippet=r.get("content") or r.get("snippet") or "",
                metadata={"score": r.get("score")},
            )
            documents.append(doc)
        return documents

    def _search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Fallback to offline corpus if available, or synthetic fallback."""
        corpus_dir = Path("ai_agent_offline_research_corpus_v2/topics")
        if not corpus_dir.exists():
            corpus_dir = (
                Path(__file__).parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"
            )

        docs: list[SourceDocument] = []
        if corpus_dir.exists():
            query_words = set(query.lower().split())
            topic_files = list(corpus_dir.glob("*.json"))

            matched_data = []
            for tf in topic_files:
                try:
                    with open(tf, encoding="utf-8") as f:
                        data = json.load(f)
                        score = sum(
                            1 for w in query_words if w in tf.stem.lower() or w in str(data).lower()
                        )
                        matched_data.append((score, data))
                except Exception:
                    continue

            matched_data.sort(key=lambda x: x[0], reverse=True)

            for _, data in matched_data:
                embedded_sources = data.get("sources", []) or data.get("source_documents", [])
                for s in embedded_sources:
                    docs.append(
                        SourceDocument(
                            title=s.get("title")
                            or s.get("source_id")
                            or "Offline Knowledge Source",
                            url=s.get("url") or f"urn:corpus:{s.get('source_id', 'doc')}",
                            snippet=s.get("snippet") or s.get("summary") or s.get("text", "")[:300],
                            metadata={"offline": True, "source_id": s.get("source_id")},
                        )
                    )
                    if len(docs) >= max_results:
                        break
                if len(docs) >= max_results:
                    break

        if not docs:
            docs = [
                SourceDocument(
                    title="GraphRAG and Multi-Agent Systems Architecture Overview",
                    url="https://example.org/research/graphrag-overview",
                    snippet=(
                        "GraphRAG combines knowledge graphs with LLM retrieval to improve "
                        "global reasoning, cross-document synthesis, and entity disambiguation."
                    ),
                    metadata={"synthetic": True},
                ),
                SourceDocument(
                    title="Comparative Evaluation: Single-Agent vs Multi-Agent Systems",
                    url="https://example.org/research/multi-agent-eval",
                    snippet=(
                        "Multi-agent architectures separate responsibilities between planning, "
                        "retrieval, critical analysis, and drafting, reducing hallucination."
                    ),
                    metadata={"synthetic": True},
                ),
                SourceDocument(
                    title="Engineering Reliable Agent Workflows with LangGraph",
                    url="https://example.org/research/reliable-agent-workflows",
                    snippet=(
                        "Stateful orchestration with explicit routing and guardrails prevents "
                        "infinite loops and ensures robust error recovery in production."
                    ),
                    metadata={"synthetic": True},
                ),
            ]
        return docs[:max_results]
