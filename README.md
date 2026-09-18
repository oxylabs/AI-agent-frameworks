# AI-agent-frameworks

[![AI Agent Frameworks: CrewAI vs LangGraph vs Claude Agent SDK](https://github.com/oxylabs/AI-agent-frameworks/blob/main/AI-agent-frameworks-banner.png)](https://oxylabs.io/web-api-early-access?utm_source=github&utm_medium=referral&utm_content=web_api_waitinglist&groupid=877)

[![](https://dcbadge.limes.pink/api/server/Pds3gBmKMH?style=for-the-badge&theme=discord)](https://discord.gg/Pds3gBmKMH) [![YouTube](https://img.shields.io/badge/YouTube-Oxylabs-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@oxylabs)

# AI Agent Frameworks in 2026 – CrewAI vs LangGraph vs Claude Agent SDK

This repository is a hands-on, side-by-side comparison of the three AI agent frameworks developers ask about most in 2026: CrewAI, LangGraph, and the Claude Agent SDK. The same agent is built three times — once per framework — so what's being compared is the framework, not the task. Each version searches Google for `best stocks to invest this year` through the [Oxylabs MCP](https://github.com/oxylabs/oxylabs-mcp) server, scrapes the top 5 results, and returns one structured digest: title, source, 4 key points, link.

Same LLM, same generation parameters, byte-identical `schema.py`, same frozen URLs. The AI agent framework is the only variable. It's built as a developer resource for anyone researching LLM frameworks, AI agents, and how CrewAI, LangGraph, and the Claude Agent SDK actually differ once you wire one up to a real, tool-using job like search-and-scrape.

- [What Are AI Agent Frameworks?](#what-are-ai-agent-frameworks)
- [Key AI Agent Framework Features](#key-ai-agent-framework-features)
- [What's in Here](#whats-in-here)
  * [Requirements](#requirements)
  * [Setup](#setup)
- [`basic` vs `extra`](#basic-vs-extra)
- [Reproducibility](#reproducibility)
- [Observability](#observability)
- [AI Agent Frameworks vs. Prompting an LLM or Using an AI Scraper](#ai-agent-frameworks-vs-prompting-an-llm-or-using-an-ai-scraper)
- [Practical Use Cases](#practical-use-cases)
- [Related Oxylabs Tooling](#related-oxylabs-tooling)
- [Learn More](#learn-more)

## What Are AI Agent Frameworks?

An AI agent framework is the orchestration layer that sits on top of an LLM and turns it into an AI agent — something that can call tools, hold state across multiple steps, and decide what to do next, instead of returning one answer to one prompt. The LLM still does the reasoning; the framework is what wires that reasoning to search, scraping, memory, approval steps, and structured output.

In 2026, three of the most widely used AI agent frameworks for this kind of tool-using workload are:

- **[CrewAI](https://docs.crewai.com/)** — a role-based framework where agents are defined as "crew" members with tasks, tools, and delegation between them.
- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** — a graph-based orchestration framework from the LangChain team, where an agent's logic is expressed as nodes and edges over explicit state.
- **[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)** — Anthropic's own SDK for building agents directly on Claude, with less framework abstraction between the developer and the model.

This repository builds the identical agent in all three: it takes a query, calls the [Oxylabs MCP](https://github.com/oxylabs/oxylabs-mcp) server to search Google and scrape the top 5 results, and returns a structured digest through a shared `schema.py` contract. Because the model, the parameters, the schema, and even the URLs being scraped are pinned identically across all three directories, what's actually being compared is the framework itself: how each one wires up tool calls, state, human approval, logging, and observability for the exact same job.

## Key AI Agent Framework Features

- **Identical task, three frameworks** — CrewAI `1.15.17`, LangGraph `1.2.11`, and Claude Agent SDK `0.2.144` each run the same **search → scrape → digest agent**.
- **Shared MCP tool layer** — all three call the same [Oxylabs MCP](https://github.com/oxylabs/oxylabs-mcp) server for Google search and scraping, so tool access isn't a variable either.
- **Byte-identical output contract** — `schema.py` defines the structured digest (title, source, 4 key points, link) and is identical across all three directories; `md5sum */schema.py` proves it.
- **Human-in-the-loop approval** — the `*_extra.py` variant of each agent pauses so a human can approve or edit the URL list before scraping.
- **Frozen test set** — `frozen-urls.json` holds the 9 candidate URLs from the original search, so every `*_extra.py` run scrapes the same pages instead of a fresh live search.
- **Run logging and hosted tracing** — every agent writes a full run log, and each framework also plugs into its own tracing UI (CrewAI AMP, LangSmith, or OpenTelemetry).
- **Per-step token usage** — the `*_extra.py` variant prints a per-step token table, so token cost is comparable across frameworks, not just wall-clock time.
- **Pinned generation settings, wired three different ways** — the same model, max output tokens, and reasoning effort across all three, but each framework exposes them through a different mechanism. Check the [Reproducibility](#reproducibility) section for more details.

## What's in Here

| Path | What it is |
| --- | --- |
| `CrewAI_agent/` | CrewAI `1.15.17` agent — `crew_basic.py` (bare) and `crew_extra.py` (+ approval, run log, per-step tokens) |
| `LangGraph_agent/` | LangGraph `1.2.11` agent — `langgraph_basic.py` and `langgraph_extra.py` |
| `Claude_Agent_SDK_agent/` | Claude Agent SDK `0.2.144` agent — `sdk_basic.py` and `sdk_extra.py` |
| `schema.py` *(inside each framework directory)* | Shared structured-output contract, byte-identical across all three |
| `frozen-urls.json` | The 9 candidate URLs from the original search, reused so `*_extra.py` runs are reproducible |
| `example-output/` | Reference output for the frozen URL set (the `1,3,4,5,7` approval selection) |
| `.env.example` | Template for the required and optional credentials, read from the repo root |

### Requirements

- **Python 3.13** — CrewAI does not install on 3.14.
- **Oxylabs Web Scraper API credentials** — free trial at [dashboard.oxylabs.io](https://dashboard.oxylabs.io/); used by all three agents via the Oxylabs MCP server.
- **Anthropic API key** — from [console.anthropic.com](https://console.anthropic.com/); budget roughly $0.25 of usage per run.
- **LangSmith account** *(optional)* — free tier at [smith.langchain.com](https://smith.langchain.com/), only needed for LangGraph's hosted tracing.

### Setup

#### Step 1 — Get the repo

```bash
git clone <REPO_URL>
cd <REPO_DIR>
cp .env.example .env
```

#### Step 2 — Add your keys

`.env` lives at the repo root — all three agents find it automatically, whatever directory you run from.

| Variable | Where to get it |
| :-- | :-- |
| `OXYLABS_USERNAME` · `OXYLABS_PASSWORD` (required) | [dashboard.oxylabs.io](https://dashboard.oxylabs.io/) — free trial |
| `ANTHROPIC_API_KEY` (required) | [console.anthropic.com](https://console.anthropic.com/) |
| `LANGSMITH_TRACING` · `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) — free tier |

#### Step 3 — Run a framework

Each directory is self-contained: its own virtualenv, its own pinned `requirements.txt`.

**CrewAI**

```bash
cd CrewAI_agent
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python crew_basic.py     # bare agent
python crew_extra.py     # + approval step, run log, per-step tokens
```

**LangGraph**

```bash
cd LangGraph_agent
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python langgraph_basic.py
python langgraph_extra.py
```

**Claude Agent SDK**

```bash
cd Claude_Agent_SDK_agent
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python sdk_basic.py
python sdk_extra.py
```

> [!TIP]
> **Windows:** activate with `.venv\Scripts\activate`.
> **uv:** swap in `uv venv --python 3.13` and `uv pip install -r requirements.txt`.

**Latest versions of the frameworks:** These install the same top-level packages, unpinned. The frameworks move fast — `requirements.txt` is what the companion video was recorded against.

| Framework | Command |
| :-- | :-- |
| CrewAI | `pip install 'crewai[anthropic]'` |
| LangGraph | `pip install langgraph langchain-anthropic langchain-mcp-adapters python-dotenv` |
| Claude Agent SDK | `pip install claude-agent-sdk pydantic python-dotenv` |


## `basic` vs `extra`

| Criteria | `*_basic.py` | `*_extra.py` |
| :-- | :-- | :-- |
| **URLs** | live search, top 5 | live search, then the frozen list from [`frozen-urls.json`](frozen-urls.json) |
| **Human-in-the-loop** | – | approve or edit the URLs before scraping |
| **Run log** | – | full trace → `run-*.log` |
| **Tracing UI** | – | full trace in LangSmith and CrewAI AMP |
| **Token usage** | – | per-step table printed at the end |

At the approval prompt in `*_extra.py`:

- **`Enter`** — take the first 5 candidates
- **`1,3,4,5,7`** — the exact set behind [`example-output/`](example-output/)

## Reproducibility

Every agent runs the same model with the same four settings:

| Setting | Value |
| :-- | :-- |
| Model | `claude-sonnet-5` |
| Max output tokens | `8000` |
| Thinking | `disabled` |
| Reasoning effort | `high` |

Getting all four onto the wire is not uniform across AI agent frameworks, which explains some of the code:

- **LangGraph** takes all four as plain `ChatAnthropic` arguments.
- **Claude Agent SDK** has no `max_tokens` option — it goes through `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.
- **CrewAI** needs the `MatchParams` httpx interceptor to patch `thinking` and `output_config` into the outgoing request body.

`schema.py` is the shared output contract, byte-identical in all three directories:

```bash
md5sum */schema.py      # macOS: md5 */schema.py
```

`frozen-urls.json` holds the 9 candidates from the original search, so every run is offered the same list. Delete it and the next `*_extra.py` run writes a fresh one from a live search.

## Observability

| Framework | In the terminal | Hosted |
| :-- | :-- | :-- |
| **CrewAI** | `verbose=True` → labelled panels in `run-crew.log` | [CrewAI AMP](https://docs.crewai.com/v1.15.17/en/observability/tracing) — `tracing=True` is already set; run `crewai login` once ([app.crewai.com](https://app.crewai.com/)) |
| **LangGraph** | `print_mode="updates"` → `run-lg.log` | [LangSmith](https://docs.langchain.com/oss/python/langgraph/observability) — set `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` in `.env` |
| **Claude Agent SDK** | message stream → `run-sdk.log` | [OpenTelemetry](https://code.claude.com/docs/en/agent-sdk/observability) — uncomment the `OTEL_*` block in `.env`; you host the collector |

> [!NOTE]
> No CrewAI AMP account? Set `tracing=False` in `run_crew()`. No LangSmith key? Set `LANGSMITH_TRACING=false`.

## AI Agent Frameworks vs. Prompting an LLM or Using an AI Scraper

AI-powered search and scraping can mean a few different things in practice. This section lays out where an AI agent framework — the approach used in this repository — sits relative to two other common patterns: prompting a general-purpose LLM directly, and an AI scraper that uses an LLM only to interpret pages during extraction.

| Criteria | AI Agent Framework (CrewAI / LangGraph / Claude Agent SDK) | Prompting a General-Purpose LLM | AI Scraper (LLM-Based Extraction) |
| --- | --- | --- | --- |
| **What orchestrates the steps** | The framework — it sequences search, scrape, and formatting as tool calls | The developer, outside the model — one call in, one answer out | The scraper's own pipeline — the LLM is invoked once per page |
| **State across steps** | Kept by the framework (crew task state, graph state, or SDK session) | None built in — each call is independent unless the developer manages it | None — each page is parsed independently |
| **Tool use** | Native — agents call MCP tools for search and scraping | Not built in — results have to be fetched and pasted into the prompt manually | The "tool" is the scrape target itself; the LLM only extracts, it doesn't decide what to fetch next |
| **Human-in-the-loop** | Supported as a step in the flow (the `*_extra.py` approval prompt in this repo) | Only possible outside the call, between separate prompts | Not part of the pattern |
| **Observability** | Framework-native tracing (CrewAI AMP, LangSmith, OpenTelemetry) | Whatever logging the developer adds around each call | Whatever logging the scraper adds around each page |
| **Ideal use** | A repeatable, multi-step task like "search, scrape, and structure a digest" | Ad hoc questions that don't need tool access or multi-step state | Turning a known page or page type into structured data |

In short: this repo is a comparison of the orchestration layer, not the underlying model — every framework calls the same LLM, the same MCP tool, and returns through the same schema.

## Practical Use Cases

1. **Picking an AI agent framework for a search-and-scrape pipeline** — teams evaluating CrewAI, LangGraph, or the Claude Agent SDK for a similar tool-using task can see how each one wires up the same MCP tool, output schema, and approval step before committing to one for production.
2. **Learning the AI agent framework workflow end to end** — because the full comparison runs for roughly $0.25 of API usage per framework, it works as a teaching example for tool calling, structured output, and human-in-the-loop design across three different LLM frameworks.
3. **Choosing an observability stack for AI agents** — the three frameworks each plug into a different tracing tool (CrewAI AMP, LangSmith, OpenTelemetry), so this repo doubles as a side-by-side look at what agent observability looks like in each one.
4. **Benchmarking framework overhead on an identical task** — with the model, schema, and URLs pinned, the only things left to vary are framework code, run logs, and token usage, which is useful for comparing AI agent frameworks on cost or complexity rather than raw model quality.

## Related Oxylabs Tooling

If you're building agents or LLM-based integrations outside this exact three-framework comparison, these related repositories are built for that:

- **[Oxylabs MCP](https://github.com/oxylabs/oxylabs-mcp)** — the MCP server used by all three agents in this repo; exposes Google search, Amazon, and general-purpose scraping tools, plus AI-powered extraction tools, to any MCP-compatible client.
- **[Oxylabs Agent Skills](https://github.com/oxylabs/agent-skills)** — official `SKILL.md` instructions covering Proxies, Web Unblocker, Web Scraper API, Headless Browser, and Video Data, installable directly into Claude Code or any other agent that supports skills, for teams that want their AI agents to reach for the right Oxylabs product without wiring up MCP.
- **[Langchain Oxylabs](https://github.com/oxylabs/langchain-oxylabs)** — a LangChain integration package for teams building LangChain agents or RAG pipelines outside LangGraph specifically, giving agents a Google-search tool and a document loader backed by the Oxylabs Web Scraper API.
- **[LLM Fine-Tuning](https://github.com/oxylabs/LLM-Fine-Tuning)** — a companion resource for going further than orchestration: it fine-tunes `Qwen3-1.7B-Base` on ~10,000 scraped Amazon listings, for when the task calls for a model trained on your own data instead of an LLM wired up through an AI agent framework.

## Learn More

- [CrewAI documentation](https://docs.crewai.com/) — reference for the crew, task, and tool concepts used in `CrewAI_agent/`.
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview) — reference for the graph and state concepts used in `LangGraph_agent/`.
- [Claude Agent SDK documentation](https://code.claude.com/docs/en/agent-sdk/overview) — reference for the SDK used in `Claude_Agent_SDK_agent/`.
- [LangSmith observability documentation](https://docs.langchain.com/oss/python/langgraph/observability) — for LangGraph's hosted tracing.
- [CrewAI AMP tracing documentation](https://docs.crewai.com/v1.15.17/en/observability/tracing) — for CrewAI's hosted tracing.
- [Claude Agent SDK observability documentation](https://code.claude.com/docs/en/agent-sdk/observability) — for the OpenTelemetry setup used by the `Claude_Agent_SDK_agent/`.

## Contact Us

If you have questions or need support, reach out at <support@oxylabs.io>, or via live chat in the [Oxylabs Dashboard](https://dashboard.oxylabs.io/). For enterprise inquiries, contact your dedicated account manager.
