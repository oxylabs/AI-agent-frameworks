"""Stock digest agent | Claude Agent SDK

One query() call runs the whole loop: the model picks Oxylabs tools until
every page is scraped, then returns the digest through the schema given to
'output_format'.
"""

import asyncio
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from dotenv import load_dotenv
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-sdk-basic.md"

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

OPTIONS = ClaudeAgentOptions(
    model="claude-sonnet-5",
    thinking={"type": "disabled"},
    effort="high",
    env={
        # max_tokens has no option, and Haiku titles the session by default.
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    },
    mcp_servers={
        "oxylabs": {
            "type": "http",
            "url": "https://mcp.oxylabs.io/mcp",
            "headers": {
                "X-Oxylabs-Username": os.environ["OXYLABS_USERNAME"],
                "X-Oxylabs-Password": os.environ["OXYLABS_PASSWORD"],
            },
        }
    },
    allowed_tools=[
        "mcp__oxylabs__google_search_scraper",
        "mcp__oxylabs__universal_scraper",
    ],
    # Allowing two tools does not stop the other eight being sent.
    disallowed_tools=[
        "mcp__oxylabs__ai_*",
        "mcp__oxylabs__amazon_*",
        "mcp__oxylabs__generate_schema",
    ],
    # Drop the 24 built-in tools, and the machine's CLAUDE.md and skills.
    tools=[],
    setting_sources=[],
    output_format={
        "type": "json_schema",
        "schema": Digest.model_json_schema(),
    },
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


async def main() -> None:
    """Run the agent and save the digest it returns."""
    prompt = f"{TASK}\n\n{SUMMARY_TASK}"
    async for message in query(prompt=prompt, options=OPTIONS):
        if isinstance(message, ResultMessage) and message.structured_output:
            digest = Digest.model_validate(message.structured_output)
            OUTPUT_FILE.write_text(to_markdown(digest.articles))


if __name__ == "__main__":
    asyncio.run(main())
