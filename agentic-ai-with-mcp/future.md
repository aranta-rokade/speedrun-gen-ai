1. Context Window Optimization (easy)
Problem: Your contents list grows every step. A 20-step project sends the full history of every file's content on every Gemini call.

What to add:

Sliding window — only send the last N messages + the original user request
Summary compaction — after every 10 steps, ask Gemini to summarize progress so far, replace history with the summary
Token counting — use client.models.count_tokens() before each call, compact when you're approaching the limit
You'll learn: tokenization, context management, prompt compression — critical for real agents.

2. Streaming Responses (easy)
Problem: You wait for the full response before printing anything. Long write_to_file responses feel like the agent is frozen.

What to add:

Use generate_content(..., stream=True) and print tokens as they arrive
Show a live spinner or progress indicator during tool execution
You'll learn: streaming APIs, async iterators, UX for long-running agents.

3. Sandboxed Execution (medium)
Problem: Your MCP tools write directly to the filesystem with no restrictions.

What to add:

Docker sandbox — new MCP tool run_command(cmd) that executes inside a Docker container
After scaffolding, the agent could pip install -r requirements.txt and python main.py to verify the project works
Restrict file operations to a specific directory (chroot-style path validation in os-mcp.py)
You'll learn: container isolation, secure tool execution, self-verifying agents.

4. RAG with Vector Embeddings (medium)
Problem: The agent generates code from its training data. It can't reference your existing codebase or docs.

What to add:

Embed a knowledge base (e.g., framework docs, your coding standards, example projects) using Gemini's embed_content() API
Store in a lightweight vector DB (ChromaDB or FAISS)
Before scaffolding, retrieve the top-K relevant snippets and inject them into the system prompt
Example: "Scaffold a FastAPI project" → RAG retrieves your team's FastAPI template patterns
You'll learn: text embeddings, similarity search, retrieval-augmented generation, vector databases.

5. Multi-Agent Architecture (medium-hard)
Problem: One agent does everything — architecture, file creation, code writing.

What to add:

Architect agent — plans the folder structure (returns JSON plan)
Coder agent — writes the actual file contents
Reviewer agent — reads generated files and suggests improvements
Agents communicate via a shared message bus or sequential handoff
You'll learn: agent orchestration, task decomposition, inter-agent communication.

6. Guardrails & Validation (medium)
Problem: The agent can write anything — broken code, malicious scripts, empty files.

What to add:

Post-step validation — after write_to_file, run a linter (ruff, eslint) via MCP tool and feed errors back
Content filtering — reject files that are suspiciously short or contain placeholder text like "TODO"
Schema validation — validate package.json, Dockerfile, etc. against known schemas
You'll learn: output validation, automated quality checks, feedback-driven self-correction (makes it more agentic).

7. Persistent Memory with Embeddings (hard)
Problem: Each run is stateless. The agent doesn't learn from past scaffolding runs.

What to add:

After each successful run, embed the project structure + summary into a vector store
On new runs, retrieve similar past projects: "Last time you scaffolded a REST API, you used this structure..."
Store user feedback ("I liked this structure" / "don't use Flask, use FastAPI")
You'll learn: long-term memory, preference learning, personalization.