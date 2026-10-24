# pip install -q -U google-genai python-dotenv mcp

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from mcp import ClientSession
from mcp.client.sse import sse_client

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MODEL_NAME: str = "gemini-2.5-flash"
MCP_SERVER_URL: str = "http://localhost:8000/sse"

MAX_STEPS: int = 50                  # max agentic loop iterations
REQUEST_DELAY: int = 7               # seconds between Gemini calls (free-tier: 10 req/min)
MAX_RETRIES: int = 3                 # retries on rate-limit / server-busy errors
COMPRESS_FREQ: int = 8               # compress context every N tool calls
NUDGE_THRESHOLD: int = 5             # steps before allowing a text-only (done) reply
BACKOFF_BASE: int = 30               # base seconds for exponential backoff
MCP_TOOL_TIMEOUT: float = 30.0       # seconds before a tool call is considered stuck
LLM_TEMPERATURE: float = 0.7         # creativity of Gemini responses (0.0 = deterministic, 1.0 = very creative)

ENV_STR_GEMINI_API_KEY: str = "GEMINI_API_KEY"
ENV_STR_OUTPUT_DIR: str = "OUTPUT_DIR"


# ── Supported project styles ───────────────────────────────────────────
PROJECT_STYLES: dict[str, str] = {
    "rest-api":          "REST API backend (Flask/FastAPI) with routes, models, middleware",
    "cli-tool":          "Command-line tool with argument parsing and subcommands",
    "web-app":           "Frontend web app (HTML/CSS/JS or React/Vue)",
    "fullstack":         "Full-stack app with frontend + backend + database",
    "data-science":      "Data science project with notebooks, data, and scripts",
    "ml-pipeline":       "Machine learning pipeline with training, evaluation, and inference",
    "python-package":    "Publishable Python package with setup.py, tests, and docs",
    "chrome-extension":  "Chrome browser extension with manifest, popup, and background scripts",
    "discord-bot":       "Discord bot with commands, events, and cogs",
    "mobile-app":        "Mobile app (React Native / Flutter) with screens and navigation",
    "microservice":      "Microservice with Docker, health checks, and message queue support",
    "static-site":       "Static site / blog with templates and content",
    "game":              "Simple game project (Pygame / JS Canvas) with assets and scenes",
    "vscode-extension":  "VS Code extension with commands, views, and settings",
    "terraform":         "Infrastructure-as-Code project with Terraform modules and envs",
}


# ── System prompt (zero-shot) ─────────────────────────────────────────
SYSTEM_PROMPT: str = """\
You are an expert software architect agent. You scaffold projects ONE STEP AT A \
TIME using the provided tools.

Rules:
- ALWAYS call exactly ONE tool per turn — never reply with just text unless you \
are completely finished
- Start by creating the root project folder, then build out subfolders and files
- Write REAL, functional starter code — not placeholders or TODOs
- Include config files (.gitignore, Dockerfile, etc.) relevant to the project style
- Include a README.md with setup instructions
- Include requirements.txt or package.json with real dependencies
- Use best practices for the given project style
- ONLY when ALL files are created, respond with a brief text summary (no tool \
call) listing every file you created
- Do NOT explain what you will do — just do it by calling a tool
"""


# ── Helpers ────────────────────────────────────────────────────────────

def _validate_env(*required: str) -> dict[str, str]:
    """Validate that required environment variables are set.
    Args:
        *required: Names of environment variables that must be present.
    Returns:
        A dict mapping each variable name to its value.
    Raises:
        SystemExit: If any required variable is missing.
    """
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Set them in a .env file or export them in your shell.")
        sys.exit(1)
    return {var: os.environ[var] for var in required}


def _backoff(attempt: int) -> float:
    """Calculate exponential backoff with full jitter.
    Args:
        attempt: The current retry attempt (1-based).
    Returns:
        Seconds to wait before the next retry.
    """
    return random.uniform(0, BACKOFF_BASE * (2 ** (attempt - 1)))


# ── MCP verification ──────────────────────────────────────────────────

async def verify_mcp() -> None:
    """Connect to the MCP server and list all available tools."""
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            logger.info("MCP tools discovered: %s", [t.name for t in mcp_tools])
            for t in mcp_tools:
                logger.info("  %s: %s", t.name, t.description)
                logger.debug("    Parameters: %s", dict(t.inputSchema.items()))

    logger.info("MCP server verified and ready")


# ── Context compression ───────────────────────────────────────────────

def compress_context(contents: list[types.Content]) -> list[types.Content]:
    """Build a programmatic summary from tool-call history and compact the conversation.
    Replaces all intermediate messages with a structured summary, keeping:
      - The original user request
      - A bullet-list summary of every tool call
      - The last two messages (for continuity)
    No LLM call is needed — the summary is deterministic and 100 % accurate.
    Args:
        contents: The full conversation history.
    Returns:
        A shortened conversation list.
    """
    summary_lines: list[str] = []
    for msg in contents[1:]:  # skip original request
        for part in msg.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                if fc.name == "write_to_file":
                    summary_lines.append(f"- Wrote {fc.args.get('file_path', '?')}")
                elif fc.name == "create_folder":
                    summary_lines.append(f"- Created folder {fc.args.get('folder_path', '?')}")
                elif fc.name == "create_file":
                    summary_lines.append(f"- Created file {fc.args.get('file_path', '?')}")

    summary_text = "Files and folders created so far:\n" + "\n".join(summary_lines)

    compressed = [
        contents[0],  # original user request (style + topic)
        types.Content(role="model", parts=[
            types.Part.from_text(text=summary_text),
        ]),
        types.Content(role="user", parts=[
            types.Part.from_text(text="Continue scaffolding. Call the next tool."),
        ]),
    ] + contents[-2:]  # keep the last tool call + result for continuity

    return compressed


# ── Orchestrator — agentic loop ───────────────────────────────────────

async def scaffold_project(
    client: genai.Client,
    style: str,
    topic: str,
    output_dir: str,
) -> None:
    """Scaffold a full project by orchestrating Gemini + MCP in an agentic loop.
    Connects to the MCP server, discovers tools, 
    then enters a Think → Act → Observe loop 
    where Gemini decides one tool call at a time until the project
    is complete or ``MAX_STEPS`` has reached.

    Args:
        client: An authenticated ``genai.Client``.
        style:  A key from ``PROJECT_STYLES``.
        topic:  A free-text description of the project to create.
        output_dir: Absolute path to the directory where the project is written.
    """
    if style not in PROJECT_STYLES:
        logger.error("Unknown style '%s'. Choose from: %s", style, ", ".join(PROJECT_STYLES))
        return

    os.makedirs(output_dir, exist_ok=True)

    tool_call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # ── Connect to MCP server ─────────────────────────────────────────
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover mcp tools & convert to Gemini function declarations
            mcp_tools = (await session.list_tools()).tools
            gemini_tools = types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters={
                        k: v for k, v in tool.inputSchema.items()
                        if k not in ("additionalProperties", "$schema")
                    },
                )
                for tool in mcp_tools
            ])

            # Start conversation
            contents: list[types.Content] = [
                types.Content(role="user", parts=[
                    types.Part.from_text(text=(
                        f'Create a **{style}** project about: "{topic}"\n'
                        f'Style description: {PROJECT_STYLES[style]}\n'
                        f'Scaffold the project step by step.'
                    )),
                ]),
            ]

            logger.info("Agent started: %s project — \"%s\"", style, topic)
            logger.info("Output dir: %s\n", output_dir)

            # ── Agentic loop ──────────────────────────────────────────
            for step in range(1, MAX_STEPS + 1):

                # THINK: ask Gemini 
                # to take the next step by calling exactly one tool, 
                # with retries
                response = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        await asyncio.sleep(REQUEST_DELAY)
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                tools=[gemini_tools],
                                temperature=LLM_TEMPERATURE,
                            ),
                        )
                        break  # success
                    except ClientError as exc:
                        err = str(exc)
                        is_retryable = ("429" in err or "503" in err)
                        if is_retryable and attempt < MAX_RETRIES:
                            # exponential-backoff retry
                            wait = _backoff(attempt)
                            label = "RATE LIMIT" if "429" in err else "SERVER BUSY"
                            logger.warning(
                                "  Step %d: [%s] Waiting %.1fs (attempt %d/%d)...",
                                step, label, wait, attempt, MAX_RETRIES,
                            )
                            await asyncio.sleep(wait)
                        else:
                            raise

                # ── Token tracking ────────────────────────────────────
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    total_prompt_tokens += getattr(usage, "prompt_token_count", 0)
                    total_completion_tokens += getattr(usage, "candidates_token_count", 0)

                # Guard against empty / blocked responses
                candidate = response.candidates[0] if response.candidates else None
                if not candidate or not candidate.content or not candidate.content.parts:
                    reason = getattr(candidate, "finish_reason", "unknown") if candidate else "no candidates"
                    logger.warning("  Step %d: [SKIP] Empty response (reason: %s), nudging model...", step, reason)
                    contents.append(types.Content(role="user", parts=[
                        types.Part.from_text(text="Continue with the next step."),
                    ]))
                    continue

                contents.append(candidate.content)
                part = candidate.content.parts[0]

                # DONE? (text reply = no more tool calls)
                fc = getattr(part, "function_call", None)
                if not fc or not fc.name:
                    if step < NUDGE_THRESHOLD:
                        logger.info("  Step %d: [NUDGE] Text reply too early, pushing to act...", step)
                        contents.append(types.Content(role="user", parts=[
                            types.Part.from_text(text="Don't explain — call a tool now. Start creating the project."),
                        ]))
                        continue
                    logger.info("\n[DONE] Agent finished in %d steps!", step)
                    logger.info("  %s", part.text)
                    logger.info("  Project at: %s", output_dir)
                    logger.info(
                        "  Tokens — prompt: %d, completion: %d, total: %d",
                        total_prompt_tokens, total_completion_tokens,
                        total_prompt_tokens + total_completion_tokens,
                    )
                    return

                # ACT: execute tool via MCP (with timeout)
                fn_name: str = fc.name
                fn_args: dict[str, Any] = dict(fc.args) if fc.args else {}

                try:
                    if fn_name == "write_to_file":
                        logger.info("  Step %d: write_to_file('%s')", step, fn_args.get("file_path", ""))
                    else:
                        arg_val = next(iter(fn_args.values()), "")
                        logger.info("  Step %d: %s('%s')", step, fn_name, arg_val)

                    result = await asyncio.wait_for(
                        session.call_tool(fn_name, fn_args),
                        timeout=MCP_TOOL_TIMEOUT,
                    )
                    result_text = result.content[0].text if result.content else "Done"
                    logger.info("           > %s", result_text)
                except asyncio.TimeoutError:
                    result_text = f"ERROR: Tool '{fn_name}' timed out after {MCP_TOOL_TIMEOUT}s"
                    logger.error("           [TIMEOUT] %s", result_text)
                except Exception as exc:
                    result_text = f"ERROR: {exc}"
                    logger.error("           [ERR] %s", exc)

                # OBSERVE: feed result back to Gemini
                contents.append(types.Content(
                    parts=[types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result_text},
                    )],
                ))

                # COMPRESS CONTEXT every N tool calls
                tool_call_count += 1
                if tool_call_count % COMPRESS_FREQ == 0:
                    before = len(contents)
                    contents = compress_context(contents)
                    logger.info(
                        "           [COMPRESS] %d msgs → %d (after %d tool calls)",
                        before, len(contents), tool_call_count,
                    )

            logger.warning("[WARN] Agent hit max steps (%d).", MAX_STEPS)
            logger.info(
                "  Tokens — prompt: %d, completion: %d, total: %d",
                total_prompt_tokens, total_completion_tokens,
                total_prompt_tokens + total_completion_tokens,
            )


# ── Entrypoint ─────────────────────────────────────────────────────────

async def main() -> None:
    """Validate environment, verify MCP, and run the scaffolding agent."""
    load_dotenv()
    env = _validate_env(ENV_STR_GEMINI_API_KEY, ENV_STR_OUTPUT_DIR)

    client = genai.Client(api_key=env[ENV_STR_GEMINI_API_KEY])
    logger.info("MCP server URL: %s", MCP_SERVER_URL)

    await verify_mcp()
    await scaffold_project(client, "rest-api", "a bookstore inventory management system", env[ENV_STR_OUTPUT_DIR])

    # ── More examples (uncomment any to try) ──────────────────────────
    # await scaffold_project(client, "cli-tool",          "a file duplicate finder",             output_dir)
    # await scaffold_project(client, "fullstack",         "a todo app with authentication",      output_dir)
    # await scaffold_project(client, "ml-pipeline",       "sentiment analysis on product reviews", output_dir)
    # await scaffold_project(client, "discord-bot",       "a music trivia quiz bot",             output_dir)
    # await scaffold_project(client, "chrome-extension",  "a website reading time estimator",    output_dir)
    # await scaffold_project(client, "microservice",      "a URL shortener service",             output_dir)
    # await scaffold_project(client, "data-science",      "analyzing global CO2 emissions",      output_dir)
    # await scaffold_project(client, "python-package",    "a markdown-to-HTML converter library", output_dir)
    # await scaffold_project(client, "terraform",         "AWS ECS deployment with ALB",         output_dir)
    # await scaffold_project(client, "game",              "a snake game with power-ups",         output_dir)
    # await scaffold_project(client, "vscode-extension",  "a code snippet manager",              output_dir)
    # await scaffold_project(client, "mobile-app",        "a habit tracker with streaks",        output_dir)
    # await scaffold_project(client, "static-site",       "a personal developer portfolio",      output_dir)


if __name__ == "__main__":
    asyncio.run(main())
