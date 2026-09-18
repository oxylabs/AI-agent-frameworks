"""Stock digest agent | Claude Agent SDK | With HITL, logs, token usage

Same agent as sdk_basic.py plus an approval step after the search, a frozen
candidate list so every run is offered the same URLs, a run log, and the
token counts of every model call.
"""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from dotenv import load_dotenv
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-sdk-extra.md"
LOG_FILE = Path(__file__).parent / "run-sdk.log"
FROZEN_FILE = Path(__file__).parent.parent / "frozen-urls.json"

PAGES = 5

STEPS = {
    "mcp__oxylabs__google_search_scraper": "search",
    "mcp__oxylabs__universal_scraper": "scrape",
    "StructuredOutput": "summarize",
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


OPTIONS = ClaudeAgentOptions(
    model="claude-sonnet-5",
    thinking={"type": "disabled"},
    effort="high",
    env={
        # No built-in max_tokens option.
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8000",
        # Haiku titles the session by default, turn off extra call.
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
)


def load_candidates(messages: list) -> list[dict]:
    """Read the frozen candidate list, writing it on the very first run."""
    if FROZEN_FILE.exists():
        return json.loads(FROZEN_FILE.read_text())

    results = [
        block
        for message in messages
        if isinstance(message, UserMessage)
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    # The scraper's reply arrives as JSON wrapped in JSON.
    reply = json.loads(json.loads(str(results[-1].content))["result"])
    found = [
        {"url": item["url"], "source": item["favicon_text"]}
        for item in reply["results"]["organic"]
    ]
    FROZEN_FILE.write_text(json.dumps(found, indent=2))
    return found


def approve_urls(messages: list) -> list[dict]:
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


def usage_row(step: str, usage: dict, output: str) -> str:
    """Lay one usage record out as a row of columns."""
    return (
        f"{step:<11}{usage['input_tokens']:>8,}"
        f"{usage['cache_creation_input_tokens']:>14,}"
        f"{usage['cache_read_input_tokens']:>13,}{output:>9}"
    )


def print_usage(messages: list) -> None:
    """Print what every model call cost, a row per step."""
    calls: dict[str | None, dict] = {}
    for message in messages:
        if not isinstance(message, AssistantMessage) or not message.usage:
            continue

        # Parallel tool calls arrive as separate messages sharing one id.
        call = calls.setdefault(
            message.message_id, {"usage": message.usage, "tools": []}
        )
        call["tools"] += [
            STEPS.get(block.name, block.name)
            for block in message.content
            if isinstance(block, ToolUseBlock)
        ]

    print(
        f"\n{'step':<11}{'input':>8}{'cache write':>14}"
        f"{'cache read':>13}{'output':>9}"
    )
    for call in calls.values():
        # Output tokens are a placeholder until the turn ends.
        step = ", ".join(dict.fromkeys(call["tools"])) or "reply"
        print(usage_row(step, call["usage"], "-"))

    total = 0.0
    run = dict.fromkeys(
        (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        ),
        0,
    )
    turns = [m for m in messages if isinstance(m, ResultMessage)]
    for number, message in enumerate(turns, start=1):
        if not message.usage:
            continue

        # Each turn is its own query(), so each cost stands alone.
        cost = message.total_cost_usd or 0
        total += cost
        for column in run:
            run[column] += message.usage[column]

        output = f"{message.usage['output_tokens']:,}"
        print(
            usage_row(f"turn {number}", message.usage, output)
            + f"   ${cost:.4f}"
        )

    print(
        usage_row("run total", run, f"{run['output_tokens']:,}")
        + f"   ${total:.4f}"
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


async def run_step(
    prompt: str, options: ClaudeAgentOptions, log: TextIO
) -> list:
    """Run one step of the agent, logging every message it streams back."""
    messages = []
    async for message in query(prompt=prompt, options=options):
        print(message, file=log)
        messages.append(message)

    return messages


async def main() -> None:
    """Search, pause for the user's picks, scrape, then report the cost."""
    with LOG_FILE.open("w") as log:
        found = await run_step(SEARCH_TASK, OPTIONS, log)
        session = next(
            m.session_id for m in found if isinstance(m, ResultMessage)
        )

        picked = approve_urls(found)
        pages = "\n".join(f"{p['source']} - {p['url']}" for p in picked)
        # Given the schema up front, the model scrapes before approval.
        options = replace(
            OPTIONS,
            resume=session,
            output_format={
                "type": "json_schema",
                "schema": Digest.model_json_schema(),
            },
        )
        written = await run_step(
            f"{SCRAPE_TASK.format(pages=pages)}\n\n{SUMMARY_TASK}",
            options,
            log,
        )

    for message in written:
        if isinstance(message, ResultMessage) and message.structured_output:
            digest = Digest.model_validate(message.structured_output)
            OUTPUT_FILE.write_text(to_markdown(digest.articles))

    print_usage(found + written)


if __name__ == "__main__":
    asyncio.run(main())