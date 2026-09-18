"""Stock digest agent | Claude Agent SDK | With HITL, logs, token usage

Same agent as sdk_basic.py plus an approval step after the search, a frozen
candidate list so every run is offered the same URLs, a run log, and the
token counts of every model call.
"""

import asyncio
import json
import os
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    ToolUseBlock,
    query,
)
from dotenv import load_dotenv
from schema import Article, Digest

load_dotenv()

OUTPUT_FILE = Path(__file__).parent / "result-sdk-hitl.md"
LOG_FILE = Path(__file__).parent / "run-sdk-hitl.log"
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


def load_candidates(response: object) -> list[dict]:
    """Read the frozen candidate list, writing it on the very first run."""
    if FROZEN_FILE.exists():
        return json.loads(FROZEN_FILE.read_text())

    # The hook receives the reply as a string of JSON wrapped in JSON.
    reply = json.loads(json.loads(str(response))["result"])
    found = [
        {"url": item["url"], "source": item["favicon_text"]}
        for item in reply["results"]["organic"]
    ]
    FROZEN_FILE.write_text(json.dumps(found, indent=2))
    return found


async def approve_urls(
    data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    """Let the user pick the pages, then hand the agent its next task."""
    try:
        found = load_candidates(data.get("tool_response"))
        listing = "\n".join(
            f"{number}. {item['source']}  {item['url']}"
            for number, item in enumerate(found, start=1)
        )
        # Plain input() would block the loop reading the agent's output.
        answer = await asyncio.to_thread(
            input, f"\n{APPROVE_TASK}{listing}\n> "
        )
        if answer.strip():
            picked = [found[int(number) - 1] for number in answer.split(",")]
        else:
            picked = found[:PAGES]
    except Exception as error:
        # A hook that raises is only logged. Without this the run carries
        # on and writes a digest out of the search results alone.
        return {"continue_": False, "stopReason": f"approval failed: {error}"}

    pages = "\n".join(f"{p['source']} - {p['url']}" for p in picked)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"{SCRAPE_TASK.format(pages=pages)}\n\n{SUMMARY_TASK}"
            ),
        }
    }


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
    # The whole point: the approval rides back on the search's tool result.
    hooks={
        "PostToolUse": [
            HookMatcher(
                matcher="mcp__oxylabs__google_search_scraper",
                hooks=[approve_urls],
            )
        ]
    },
)


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


async def main() -> None:
    """Run the agent in one call, pausing inside the hook to approve."""
    messages = []
    with LOG_FILE.open("w") as log:
        async for message in query(prompt=SEARCH_TASK, options=OPTIONS):
            print(message, file=log)
            messages.append(message)

    for message in messages:
        if isinstance(message, ResultMessage) and message.structured_output:
            digest = Digest.model_validate(message.structured_output)
            OUTPUT_FILE.write_text(to_markdown(digest.articles))

    print_usage(messages)


if __name__ == "__main__":
    asyncio.run(main())
