"""Stock digest agent | CrewAI

A crew of one agent and one task: the agent loops over the Oxylabs MCP
tools until every page is scraped, then answers in the digest schema.
"""

import json
import os
from pathlib import Path
from typing import cast

import httpx
from crewai import LLM, Agent, Crew, CrewOutput, Process, Task
from crewai.llms.hooks import BaseInterceptor
from crewai.mcp import MCPServerHTTP
from crewai.mcp.filters import create_static_tool_filter
from dotenv import load_dotenv
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-crew-basic.md"

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
        headers.pop("content-length")  # httpx sets it from the new body

        return httpx.Request(
            message.method, message.url, headers=headers, json=body
        )

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Leave replies as they arrive."""
        return message


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

digest_task = Task(
    description=TASK,
    expected_output=SUMMARY_TASK,
    agent=researcher,
    output_pydantic=Digest,
)


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


def main() -> None:
    """Run the crew and save the digest it returns."""
    crew = Crew(
        agents=[researcher],
        tasks=[digest_task],
        process=Process.sequential,
    )
    # kickoff() is typed as possibly streaming, which it never is here.
    result = cast(CrewOutput, crew.kickoff())
    digest = Digest.model_validate(result.pydantic)

    OUTPUT_FILE.write_text(to_markdown(digest.articles))


if __name__ == "__main__":
    main()
