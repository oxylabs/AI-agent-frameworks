"""Stock digest agent | LangGraph

The agent loops between the model and the Oxylabs MCP tools until every page
has been scraped, then one last node writes the structured digest.
"""

import asyncio
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-lg-basic.md"

MODEL_PARAMS = {
    "model": "claude-sonnet-5",
    "max_tokens": 8_000,
    "thinking": {"type": "disabled"},
    "output_config": {"effort": "high"},
}

PAGES = 5

TASK = (
    "Search Google for 'best stocks to invest this year', passing "
    "'parse' as true so the results come back parsed.\n\n"
    f"Take the first {PAGES} organic results and scrape each URL with the "
    "universal scraper, passing 'output_format' as 'md' to get markdown "
    "back. If a page comes back empty, or with something other than the "
    "article itself, scrape it once more with 'render' set to 'html'. If "
    "that still does not work, leave that page and carry on.\n\n"
    f"Stay on those {PAGES} links. Never follow a link you find inside a "
    "scraped page."
)

SUMMARY_TASK = (
    "Now write up the pages you scraped, keeping them in the order you "
    "scraped them. Give one entry per page."
)

llm = ChatAnthropic(**MODEL_PARAMS)


class State(MessagesState):
    """The conversation so far, plus the digest once it is written."""

    digest: Digest


def build_graph(tools: list[BaseTool]):
    """Build the agent loop and the node that writes the digest."""
    llm_with_tools = llm.bind_tools(tools)
    writer = llm.with_structured_output(Digest, method="json_schema")

    async def agent(state: State) -> dict:
        """Let the model pick the next Oxylabs tool call, or stop."""
        reply = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [reply]}

    def should_continue(state: State) -> Literal["tools", "summarize"]:
        """Keep looping for as long as the model asks for tools."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "summarize"

    def enough_pages(state: State) -> Literal["agent", "summarize"]:
        """Leave the loop once every page has content, or after one retry."""
        scraped = [
            message
            for message in state["messages"]
            if getattr(message, "name", None) == "universal_scraper"
        ]
        filled = [page for page in scraped if str(page.text).strip()]

        done = len(filled) >= PAGES
        retried = len(scraped) > PAGES
        if done or retried:
            return "summarize"
        return "agent"

    async def summarize(state: State) -> dict:
        """Write up everything the agent scraped."""
        task = HumanMessage(SUMMARY_TASK)
        reply = await writer.ainvoke([*state["messages"], task])
        return {"digest": Digest.model_validate(reply)}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, ["tools", "summarize"]
    )
    builder.add_conditional_edges(
        "tools", enough_pages, ["agent", "summarize"]
    )
    builder.add_edge("summarize", END)
    return builder.compile()


def to_markdown(articles: list[Article]) -> str:
    """Lay the articles out the same way for every framework."""
    sections = []
    for article in articles:
        points = "\n".join(f"- {point}" for point in article.key_points)
        sections.append(
            f"## {article.title}\n\n"
            f"**Source:** {article.source}\n\n"
            f"**Key points:**\n\n{points}\n\n"
            f"**Link:** {article.link}"
        )

    return "# Digest\n\n" + "\n\n".join(sections)


async def main() -> None:
    """Connect to Oxylabs, run the agent and save the digest."""
    connection = StreamableHttpConnection(
        transport="streamable_http",
        url="https://mcp.oxylabs.io/mcp",
        headers={
            "X-Oxylabs-Username": os.environ["OXYLABS_USERNAME"],
            "X-Oxylabs-Password": os.environ["OXYLABS_PASSWORD"],
        },
    )
    client = MultiServerMCPClient({"oxylabs": connection})
    tools = {tool.name: tool for tool in await client.get_tools()}

    graph = build_graph(
        [tools["google_search_scraper"], tools["universal_scraper"]]
    )
    initial = State(
        messages=[HumanMessage(TASK)],
        digest=Digest(articles=[]),
    )
    result = await graph.ainvoke(initial)

    OUTPUT_FILE.write_text(to_markdown(result["digest"].articles))


if __name__ == "__main__":
    asyncio.run(main())
