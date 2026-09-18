"""Stock digest agent | LangGraph | With HITL, logs, token usage

Same agent as langgraph_basic.py plus an approval step after the search, a frozen
candidate list so every run is offered the same URLs, a run log, and the
token counts of every model call.
"""

import asyncio
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-lg-extra.md"
LOG_FILE = Path(__file__).parent / "run-lg.log"
FROZEN_FILE = Path(__file__).parent.parent / "frozen-urls.json"

MODEL_PARAMS = {
    "model": "claude-sonnet-5",
    "max_tokens": 8_000,
    "thinking": {"type": "disabled"},
    "output_config": {"effort": "high"},
    # LangChain never caches on its own.
    "model_kwargs": {"cache_control": {"type": "ephemeral"}},
}

PAGES = 5

STEPS = {
    "google_search_scraper": "search",
    "universal_scraper": "scrape",
}

SEARCH_TASK = (
    "Search Google for 'best stocks to invest this year', passing "
    "'parse' as true so the results come back parsed."
)

SCRAPE_TASK = (
    "Scrape each of these URLs with the universal scraper, passing "
    "'output_format' as 'md' to get markdown back. If a page comes back "
    "empty, or with something other than the article itself, scrape it "
    "once more with 'render' set to 'html'. If that still does not "
    "work, leave that page and carry on.\n\n"
    "Scrape only these, and never follow a link you find inside a scraped "
    "page:\n\n{pages}"
)

SUMMARY_TASK = (
    "Now write up the pages you scraped, keeping them in the order you "
    "scraped them. Give one entry per page."
)

APPROVE_TASK = (
    f"Here is what the search found - approve the first {PAGES}, or edit "
    "the list.\n"
    "Press Enter to approve, or type the numbers you want (e.g. 1,2,3,6,7)\n"
)

llm = ChatAnthropic(**MODEL_PARAMS)


class State(MessagesState):
    """The conversation, how many pages to scrape, and the digest."""

    wanted: int
    digest: dict


def load_candidates(message: BaseMessage) -> list[dict]:
    """Read the frozen candidate list, writing it on the very first run."""
    if FROZEN_FILE.exists():
        return json.loads(FROZEN_FILE.read_text())

    results = json.loads(str(message.text))["results"]["organic"]
    found = [
        {"url": item["url"], "source": item["favicon_text"]}
        for item in results
    ]
    FROZEN_FILE.write_text(json.dumps(found, indent=2))
    return found


def approve_urls(message: BaseMessage) -> list[dict]:
    """Show the candidates and let the user approve or edit the list."""
    found = load_candidates(message)
    listing = "\n".join(
        f"{number}. {item['source']}  {item['url']}"
        for number, item in enumerate(found, start=1)
    )
    answer = interrupt(APPROVE_TASK + listing)

    if answer.strip():
        return [found[int(number) - 1] for number in answer.split(",")]
    return found[:PAGES]


def build_graph(tools: list[BaseTool]):
    """Build the agent loop, the approval step and the digest writer."""
    llm_with_tools = llm.bind_tools(tools)
    writer = llm.with_structured_output(
        Digest, method="json_schema", include_raw=True
    )

    async def agent(state: State) -> dict:
        """Let the model pick the next Oxylabs tool call, or stop."""
        reply = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [reply]}

    def review(state: State) -> dict:
        """Swap the search results for the pages the user approved."""
        last = state["messages"][-1]
        if getattr(last, "name", None) != "google_search_scraper":
            return {}

        picked = approve_urls(last)
        pages = "\n".join(f"{p['source']} - {p['url']}" for p in picked)
        task = SCRAPE_TASK.format(pages=pages)
        return {"messages": [HumanMessage(task)], "wanted": len(picked)}

    def enough_pages(state: State) -> Literal["agent", "summarize"]:
        """Leave the loop once every page has content, or after one retry."""
        scraped = [
            message
            for message in state["messages"]
            if getattr(message, "name", None) == "universal_scraper"
        ]
        filled = [page for page in scraped if str(page.text).strip()]

        done = len(filled) >= state["wanted"]
        retried = len(scraped) > state["wanted"]
        if done or retried:
            return "summarize"
        return "agent"

    def should_continue(state: State) -> Literal["tools", "summarize"]:
        """Keep looping for as long as the model asks for tools."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "summarize"

    async def summarize(state: State) -> dict:
        """Write up everything the agent scraped."""
        task = HumanMessage(SUMMARY_TASK)
        reply = cast(
            dict[str, Any], await writer.ainvoke([*state["messages"], task])
        )
        return {
            "digest": reply["parsed"].model_dump(),
            "messages": [reply["raw"]],
        }

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("review", review)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, ["tools", "summarize"]
    )
    builder.add_edge("tools", "review")
    builder.add_conditional_edges(
        "review", enough_pages, ["agent", "summarize"]
    )
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=InMemorySaver())


def split_usage(usage: UsageMetadata) -> dict[str, int]:
    """Split one call's tokens into the four columns."""
    # Anthropic books cache writes under an undeclared 'ephemeral_5m' key.
    details = cast(dict[str, int], usage.get("input_token_details", {}))
    written = details.get("ephemeral_5m_input_tokens", 0)
    read = details.get("cache_read", 0)
    return {
        # LangChain's input_tokens is the total, cached tokens included.
        "input": usage["input_tokens"] - written - read,
        "write": written,
        "read": read,
        "output": usage["output_tokens"],
    }


def usage_row(step: str, tokens: dict[str, int]) -> str:
    """Lay one usage record out as a row of columns."""
    return (
        f"{step:<11}{tokens['input']:>8,}{tokens['write']:>14,}"
        f"{tokens['read']:>13,}{tokens['output']:>9,}"
    )


def print_usage(messages: list) -> None:
    """Print what every model call cost, a row per step."""
    total = {"input": 0, "write": 0, "read": 0, "output": 0}
    print(
        f"\n{'step':<11}{'input':>8}{'cache write':>14}"
        f"{'cache read':>13}{'output':>9}"
    )

    for message in messages:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue

        names = dict.fromkeys(
            STEPS.get(call["name"], call["name"])
            for call in message.tool_calls
        )
        if names:
            step = ", ".join(names)
        elif message is messages[-1]:
            step = "summarize"
        else:
            step = "reply"

        tokens = split_usage(message.usage_metadata)
        for column, count in tokens.items():
            total[column] += count
        print(usage_row(step, tokens))

    print(usage_row("run total", total))


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
    config: RunnableConfig = {"configurable": {"thread_id": "digest"}}
    inputs: Any = State(
        messages=[HumanMessage(SEARCH_TASK)],
        wanted=PAGES,
        digest={"articles": []},
    )

    with LOG_FILE.open("w") as log:
        while True:
            with redirect_stdout(log):
                await graph.ainvoke(inputs, config, print_mode="updates")

            state = await graph.aget_state(config)
            if not state.interrupts:
                break

            prompt = f"\n{state.interrupts[0].value}\n> "
            inputs = Command(resume=input(prompt))

    result = (await graph.aget_state(config)).values
    digest = Digest.model_validate(result["digest"])
    OUTPUT_FILE.write_text(to_markdown(digest.articles))
    print_usage(result["messages"])


if __name__ == "__main__":
    asyncio.run(main())
