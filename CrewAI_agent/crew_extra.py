"""Stock digest | CrewAI | With HITL, logs, token usage

Same crew as crew_basic.py plus an approval step after the search, a frozen
candidate list so every run is offered the same URLs, a run log, and the
token counts of every model call.
"""

import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO, cast

import httpx
from crewai import LLM, Agent, Crew, CrewOutput, Process, Task
from crewai.events import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.llms.hooks import BaseInterceptor
from crewai.mcp import MCPServerHTTP
from crewai.mcp.filters import create_static_tool_filter
from crewai.utilities.types import LLMMessage
from dotenv import load_dotenv
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-crew-extra.md"
LOG_FILE = Path(__file__).parent / "run-crew.log"
FROZEN_FILE = Path(__file__).parent.parent / "frozen-urls.json"

PAGES = 5

STEPS = {
    "mcp_oxylabs_io_mcp_google_search_scraper": "search",
    "mcp_oxylabs_io_mcp_universal_scraper": "scrape",
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

# Unsupported parameters via LLM().
WIRE_PARAMS = {
    "thinking": {"type": "disabled"},
    "output_config": {"effort": "high"},
}


class MatchParams(BaseInterceptor[httpx.Request, httpx.Response]):
    """Add the two model settings CrewAI cannot send by itself."""

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Merge them into the request body on its way to Anthropic."""
        body = json.loads(message.content) | WIRE_PARAMS
        headers = dict(message.headers)
        headers.pop("content-length")  # httpx sets it from the new body.

        return httpx.Request(
            message.method, message.url, headers=headers, json=body
        )

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Leave replies as they arrive."""
        return message


class CallUsage(BaseEventListener):
    """Collect every model call's tokens, which only the event bus has."""

    def __init__(self) -> None:
        """Start with an empty list, then register on the bus."""
        self.calls: list[tuple[str, dict[str, Any]]] = []
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        """Name each call after the tools it asked for, and bank its usage."""

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_call(source: Any, event: LLMCallCompletedEvent) -> None:
            blocks: list[dict[str, str]] = (
                event.response if isinstance(event.response, list) else []
            )
            steps = dict.fromkeys(
                STEPS.get(block["name"], block["name"]) for block in blocks
            )
            # from_agent is always None; a call with no role is the converter.
            other = "reply" if event.agent_role else "schema"
            step = ", ".join(steps) or other
            self.calls.append((step, event.usage or {}))


usage = CallUsage()

llm = LLM(
    model="anthropic/claude-sonnet-5",
    max_tokens=8_000,
    interceptor=MatchParams(),
)

oxylabs = MCPServerHTTP(
    url="https://mcp.oxylabs.io/mcp",
    headers={
        "X-Oxylabs-Username": os.environ["OXYLABS_USERNAME"],
        "X-Oxylabs-Password": os.environ["OXYLABS_PASSWORD"],
    },
    tool_filter=create_static_tool_filter(
        allowed_tool_names=["google_search_scraper", "universal_scraper"]
    ),
)

researcher = Agent(
    role="Investment article summarizer",
    goal="Summarize each article into four readable bullet points.",
    backstory="You report what the article says and add no picks of your own.",
    llm=llm,
    mcps=[oxylabs],
)

search_task = Task(
    description=SEARCH_TASK,
    expected_output="The organic results, each with its publisher and URL.",
    agent=researcher,
)

digest_task = Task(
    description=SCRAPE_TASK,
    expected_output=SUMMARY_TASK,
    agent=researcher,
    output_pydantic=Digest,
)


def load_candidates(messages: list[LLMMessage]) -> list[dict[str, str]]:
    """Read the frozen candidate list, writing it on the very first run."""
    if FROZEN_FILE.exists():
        return json.loads(FROZEN_FILE.read_text())

    reply = next(m for m in reversed(messages) if m.get("role") == "tool")
    results = json.loads(str(reply["content"]))["results"]["organic"]
    found = [
        {"url": item["url"], "source": item["favicon_text"]}
        for item in results
    ]
    FROZEN_FILE.write_text(json.dumps(found, indent=2))
    return found


def approve_urls(messages: list[LLMMessage]) -> list[dict[str, str]]:
    """Show the candidates and let the user approve or edit the list."""
    found = load_candidates(messages)
    listing = "\n".join(
        f"{number}. {item['source']}  {item['url']}"
        for number, item in enumerate(found, start=1)
    )
    answer = input(f"\n{APPROVE_TASK}{listing}\n> ")

    if answer.strip():
        return [found[int(number) - 1] for number in answer.split(",")]
    return found[:PAGES]


def split_usage(tokens: dict[str, Any]) -> dict[str, int]:
    """Split one call's tokens into the four columns."""
    # CrewAI reports input_tokens as the total, both cache columns included.
    read = tokens.get("cached_prompt_tokens", 0)
    written = tokens.get("cache_creation_tokens", 0)
    return {
        "input": tokens.get("input_tokens", 0) - read - written,
        "write": written,
        "read": read,
        "output": tokens.get("output_tokens", 0),
    }


def usage_row(step: str, tokens: dict[str, int]) -> str:
    """Lay one usage record out as a row of columns."""
    return (
        f"{step:<11}{tokens['input']:>8,}{tokens['write']:>14,}"
        f"{tokens['read']:>13,}{tokens['output']:>9,}"
    )


def print_usage(calls: list[tuple[str, dict[str, Any]]]) -> None:
    """Print what every model call cost, a row per step."""
    total = {"input": 0, "write": 0, "read": 0, "output": 0}
    print(
        f"\n{'step':<11}{'input':>8}{'cache write':>14}"
        f"{'cache read':>13}{'output':>9}"
    )

    for step, raw in calls:
        tokens = split_usage(raw)
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


def run_crew(
    task: Task, log: TextIO, inputs: dict[str, str] | None = None
) -> CrewOutput:
    """Run one task as its own crew, sending the verbose log to a file."""
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        tracing=True, # Enable tracing for CrewAI AMP.
    )
    with redirect_stdout(log):
        result = cast(CrewOutput, crew.kickoff(inputs=inputs))
        crewai_event_bus.flush()  # Panels are printed on worker threads.

    return result


def main() -> None:
    """Search, pause for the user's picks, scrape, then report the cost."""
    with LOG_FILE.open("w") as log:
        found = run_crew(search_task, log)

        picked = approve_urls(found.tasks_output[0].messages)
        pages = "\n".join(f"{p['source']} - {p['url']}" for p in picked)
        result = run_crew(digest_task, log, {"pages": pages})

    digest = Digest.model_validate(result.pydantic)
    OUTPUT_FILE.write_text(to_markdown(digest.articles))
    print_usage(usage.calls)


if __name__ == "__main__":
    main()
