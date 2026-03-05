# Agentic AI Demo — Project Scaffolding Agent with Gemini + MCP

An AI-powered project scaffolding agent built with **Google Gemini 2.5 Flash** and the **Model Context Protocol (MCP)**.
Give it a project style and a topic, and it autonomously generates a complete folder structure with real, functional starter code.

## What makes this Agentic AI?

This agent operates in a **true agentic loop** — not a single-shot prompt-and-execute:

- **Observe → Think → Act → Observe loop** — The agent runs iteratively, deciding one action at a time based on feedback
- **MCP Tool Server** — Tools (`create_folder`, `create_file`, `write_to_file`) are exposed via a standalone MCP server over SSE, discovered at runtime — not hardcoded
- **Gemini Function Calling** — Gemini natively calls tools using structured function calls — no JSON parsing or `eval()` needed
- **Autonomy** — Given only a style and topic, the agent independently decides everything
- **Self-correction** — Errors are fed back to the agent, which can adapt and retry
- **Memory** — Full conversation history across steps, compacted when it grows too large

> **TL;DR:** A regular AI *tells* you what to do. An agentic AI *does* it — step by step, with MCP tools and feedback.

> **Security** — The LLM never executes arbitrary code. It can only invoke declared MCP tools (`create_folder`, `create_file`, `write_to_file`) with typed arguments. Tools run in a separate MCP server process.

## Architecture

```
┌─────────────┐    function call   ┌──────────────┐    SSE / HTTP    ┌──────────────┐
│  Gemini LLM │ <───────────────── │  Agent Loop  │ ────────────────>│  MCP Server  │
│  (thinks)   │ ──────────────────>│(orchestrates)│ <────────────────│  (os-mcp.py) │
└─────────────┘       result       └──────────────┘      result      └──────────────┘
                                                                    localhost:8000/sse
```

1. MCP Server for OS level file handling tasks- allows the agent to safely run just the predefined apis. The agent wait for the tools for the tool to complete execution with timeouts to prevent infinite waits.
2. Agent uses gemini-2.5-flash model, with temperatue=0.7 (slighly creative).
3. Prompted the model using zero-shot prompting, no examples.
4. Context Window Optimization/Token Management by compressing/summary the tool calls after every N tool calls. This reduced the prompt token size significantly. No LLM calls needed for generating this summary. Gemini errors - rate limits and server capacity errors are handled using exponential back-off policy.
5. Logging

### Future Improvements and Explorations

1. Streaming Responses - no need wait for the full response from the agent before printing anything.
2. Guardrails & Validation - Verify the tasks before calling the tools, such as verify for suspicious files. Once tool is executed, check if task was successful, post tool checks, run linters, build at the end of the agentic loop, etc.
3. Sandboxed Execution for secure runs
4. RAG - to reference existing data
5. Multi-Agent Architecture
6. Persistent Memory of previously created projects by the agent.

## Setup

1. Install dependencies:

   ```bash
   pip install google-genai python-dotenv mcp
   ```

2. Set environment variables in a `.env` file:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   OUTPUT_DIR=./output
   ```

3. Start the MCP server in a terminal:

   ```bash
   python os-mcp.py
   ```

4. Run the agent in a second terminal:

   ```bash
   python agentic-ai-with-mcp.py
   ```

## Supported Styles

| Style | Description |
|-------|-------------|
| `rest-api` | REST API backend (Flask/FastAPI) with routes, models, middleware |
| `cli-tool` | Command-line tool with argument parsing and subcommands |
| `web-app` | Frontend web app (HTML/CSS/JS or React/Vue) |
| `fullstack` | Full-stack app with frontend + backend + database |
| `data-science` | Data science project with notebooks, data, and scripts |
| `ml-pipeline` | Machine learning pipeline with training, evaluation, and inference |
| `python-package` | Publishable Python package with setup.py, tests, and docs |
| `chrome-extension` | Chrome browser extension with manifest, popup, and background scripts |
| `discord-bot` | Discord bot with commands, events, and cogs |
| `mobile-app` | Mobile app (React Native / Flutter) with screens and navigation |
| `microservice` | Microservice with Docker, health checks, and message queue support |
| `static-site` | Static site / blog with templates and content |
| `game` | Simple game project (Pygame / JS Canvas) with assets and scenes |
| `vscode-extension` | VS Code extension with commands, views, and settings |
| `terraform` | Infrastructure-as-Code project with Terraform modules and envs |

## Sample Output

[output.log](./output.log)
