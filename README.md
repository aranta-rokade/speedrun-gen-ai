# speedrun-gen-ai

Generative AI 101 — from prompts to production.

This repo is a hands-on learning workspace covering:

- Foundational LLM usage
- Hugging Face and Gemini starter notebooks
- Agentic AI with MCP tools
- Retrieval-Augmented Generation (RAG) experiments

## Exercises Done

- [x] **[01-gemini-basics/](01-gemini-basics/gemini-api-starter.ipynb)** — LLM API basics with Gemini
  - API keys, first calls, token usage / billing metadata
  - Micro projects: AI content writer, bulk writer (with retries for rate limits), chatbot
  - Key lesson: LLMs are **stateless** — a chatbot only remembers context if you send the history each time

- [x] **[02-huggingface/](02-huggingface/huggingface-starter.ipynb)** — Pretrained models running locally via `transformers` pipelines
  - Text: NER, generation, summarization, question answering, sentiment, zero-shot classification, translation
  - Vision: image classification, object detection, image segmentation
  - Key lesson: small task-specific models vs. general-purpose LLM APIs

- [x] **[03-agentic-ai/](03-agentic-ai/)** — Agentic AI, built up in three steps

  - [x] **[a-proto-agent/](03-agentic-ai/a-proto-agent/proto-agentic-ai.ipynb)** — Project scaffolder, **single-shot** (no agent loop)
    - One Gemini call returns the whole project plan as JSON
    - Actions (`create_folder`, `create_file`, `write_to_file`) are run with a sandboxed `eval()`
    - Key lesson: structured output + tool execution is a *building block*, not a true agent — no feedback, no self-correction

  - [x] **[b-agentic-loop/](03-agentic-ai/b-agentic-loop/agentic-ai.ipynb)** — Project scaffolder, **true agentic loop**
    - Think → Act → Observe: one action per step, result fed back to Gemini
    - Conversation history as memory, errors trigger self-correction, `MAX_STEPS` safety limit
    - Key lesson: what turns an LLM into an agent is the feedback loop

  - [x] **[c-agentic-ai-with-mcp/](03-agentic-ai/c-agentic-ai-with-mcp/)** — Agent rebuilt with **MCP + native function calling**
    - Tools served by a separate MCP server (`os-mcp.py`) and discovered at runtime
    - Gemini function calling replaces JSON parsing and `eval()` with structured, typed tool calls — the LLM can only invoke declared tools, never run arbitrary code
    - Context-window optimization (history compaction), exponential back-off, timeouts, logging
    - Key lesson: moving from a notebook prototype to a safer .py, production-style architecture

- [x] **[04-rag/](04-rag/RAG.ipynb)** — Retrieval-Augmented Generation chatbot
  - Embeds Sample articles with `Qwen3-Embedding-0.6B`, retrieves top-5 by cosine similarity
  - Gemini answers **only** from the retrieved context
  - Key lesson: grounded answers on domain questions, and "I don't have enough data" on off-topic ones (less hallucination)

- [x] **[05-llm-web-search/](05-llm-web-search/LLM_web_search.ipynb)** — LLM + live web search
  - Searches the web with Serper, scrapes pages with BeautifulSoup, Gemini summarizes the results
  - Key lesson: giving an LLM fresh, real-world data beyond its training cutoff

## Quick Start

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (as needed per project folder):

   ```bash
   pip install -r requirements.txt            # everything
   # or just the MCP agent:
   pip install -r 03-agentic-ai/c-agentic-ai-with-mcp/requirements.txt
   ```

3. Set up API keys:

   ```bash
   cp .env.example .env   # then fill in GEMINI_API_KEY, SERPER_API_KEY, HF_TOKEN
   ```

4. Open the numbered folders in order (`01-` → `05-`, and `a` → `c` inside `03-agentic-ai/`) in VS Code / Google Colab and run the cells.

Requires Python 3.10+.

## Learning Goal

Understand the core concepts of Generative AI by building them, not just reading about them. Each exercise is a small, hands-on prototype that isolates one idea — LLM APIs, pretrained models, agents, tool calling, MCP, and RAG — so the concept is learned by seeing it work (and fail).

These are learning prototypes, not production code *yet*.
