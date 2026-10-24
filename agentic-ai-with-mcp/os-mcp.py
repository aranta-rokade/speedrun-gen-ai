"""
OS MCP Server — Filesystem Tools

Exposes filesystem tools (create_folder, create_file, write_to_file)
via the Model Context Protocol so the agentic AI can discover and call them.

Run as separate server:  python os-mcp.py
Connects on:             http://localhost:8000/sse

pip install mcp
"""

import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OS MCP")


@mcp.tool()
def create_folder(folder_path: str) -> str:
    """Create a directory and all parent directories at the given path."""
    os.makedirs(folder_path, exist_ok=True)
    return f"Folder created: {folder_path}"


@mcp.tool()
def create_file(file_path: str) -> str:
    """Create an empty file at the given path (parent directories are created automatically)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    open(file_path, "a").close()
    return f"File created: {file_path}"


@mcp.tool()
def write_to_file(file_path: str, file_content: str) -> str:
    """Write text content to a file, creating or overwriting it (parent directories are created automatically)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(file_content)
    return f"Content written to: {file_path}"


if __name__ == "__main__":
    mcp.run(transport="sse")
