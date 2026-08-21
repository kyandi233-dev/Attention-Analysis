"""MCP server exposing vision tools backed by Groq + Llama-4 Scout.

Two families of tools:
  - *_from_path   : analyze an image file on disk.
  - *_from_clipboard : read the image directly from the system clipboard.

The clipboard tools are the main reason this project exists: they let any
text-only LLM (DeepSeek, GLM, Qwen, ...) "see" an image the user just copied,
without asking them to save a file first.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any

import aiofiles
from groq import AsyncGroq
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .clipboard import ClipboardError, save_clipboard_image

SERVER_NAME = "clipboard-vision-mcp"
SERVER_VERSION = "0.1.0"
VISION_MODEL = os.environ.get(
    "VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)

# Security: only allow image files, bound the size to prevent a malicious /
# prompt-injected caller from exfiltrating arbitrary local files as base64.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
IMAGE_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",        # JPEG
    b"GIF87a",
    b"GIF89a",
    b"RIFF",                # WEBP (RIFF....WEBP)
    b"BM",                  # BMP
)


def _validate_image_path(path_str: str) -> Path:
    """Reject non-image paths and oversize files before opening them."""
    p = Path(path_str).resolve()
    if not p.is_file():
        raise ValueError(f"Not a file: {path_str}")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Refusing to read '{p.suffix}' — only image files are allowed "
            f"({', '.join(sorted(ALLOWED_EXTENSIONS))})."
        )
    size = p.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {size} bytes (max {MAX_IMAGE_BYTES})."
        )
    return p


def _validate_magic(data: bytes) -> None:
    if not any(data.startswith(m) for m in IMAGE_MAGIC_PREFIXES):
        raise ValueError("File content does not look like a supported image.")


server = Server(SERVER_NAME)


class VisionClient:
    def __init__(self, api_key: str):
        self.client = AsyncGroq(api_key=api_key)

    async def analyze(self, image_path: str, prompt: str) -> str:
        p = _validate_image_path(image_path)
        async with aiofiles.open(p, "rb") as f:
            data = await f.read()
        _validate_magic(data)
        b64 = base64.b64encode(data).decode("utf-8")

        response = await self.client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.5,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


vision_client: VisionClient | None = None


PROMPTS: dict[str, str] = {
    "analyze": "Describe this image in detail. Identify all relevant elements, "
               "context, and anything that would help someone who cannot see it.",
    "extract_text": "Extract ALL text from this image. Return only the text, "
                    "preserving layout and line breaks. No commentary.",
    "describe_ui": (
        "Analyze this UI screenshot. Describe: "
        "1) overall layout, 2) components (buttons, forms, navigation, inputs), "
        "3) visible text and labels, 4) state (errors, active tabs, modals)."
    ),
    "diagnose_error": (
        "Analyze this error screenshot. Return: "
        "1) the exact error message, 2) the likely cause, "
        "3) concrete steps to fix it, 4) how to prevent recurrence."
    ),
    "understand_diagram": (
        "Interpret this diagram. Return: "
        "1) diagram type, 2) components and their roles, "
        "3) relationships/flow, 4) the overall purpose."
    ),
    "analyze_chart": (
        "Analyze this chart. Return: "
        "1) chart type, 2) axes and labels, 3) key trends, "
        "4) notable data points, 5) insights."
    ),
    "code_from_screenshot": (
        "Extract all code from this screenshot. Return: "
        "1) language, 2) clean code in a fenced code block preserving indentation."
    ),
}


def _image_tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image file.",
                }
            },
            "required": ["image_path"],
        },
    )


def _clipboard_tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # Clipboard-first tools (the reason this project exists)
        Tool(
            name="analyze_clipboard",
            description=(
                "Read the image currently in the system clipboard and analyze it. "
                "Use this when the user says 'look at this', 'what's in my clipboard', "
                "or pastes a screenshot without providing a file path. "
                "Optional `prompt` overrides the default description request."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Custom question about the clipboard image.",
                    }
                },
                "required": [],
            },
        ),
        _clipboard_tool(
            "extract_text_from_clipboard",
            "OCR the image currently in the clipboard and return only its text.",
        ),
        _clipboard_tool(
            "describe_ui_from_clipboard",
            "Describe the UI in a clipboard screenshot (layout, components, state).",
        ),
        _clipboard_tool(
            "diagnose_error_from_clipboard",
            "Diagnose an error screenshot from the clipboard and propose fixes.",
        ),
        _clipboard_tool(
            "code_from_clipboard",
            "Extract code from a clipboard screenshot, identifying the language.",
        ),
        # File-path tools (for when the image is already saved)
        Tool(
            name="analyze_image",
            description="Analyze an image file. Provide `image_path` and optional `prompt`.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["image_path"],
            },
        ),
        _image_tool("extract_text", "OCR an image file."),
        _image_tool("describe_ui", "Describe a UI screenshot file."),
        _image_tool("diagnose_error", "Diagnose an error screenshot file."),
        _image_tool("understand_diagram", "Interpret a diagram image file."),
        _image_tool("analyze_chart", "Analyze a chart image file."),
        _image_tool("code_from_screenshot", "Extract code from a screenshot file."),
    ]


async def _run(prompt_key: str, image_path: str, override: str | None = None) -> str:
    assert vision_client is not None
    prompt = override or PROMPTS[prompt_key]
    return await vision_client.analyze(image_path, prompt)


async def _run_clipboard(prompt_key: str, override: str | None = None) -> str:
    try:
        path = save_clipboard_image()
    except ClipboardError as e:
        return f"Clipboard error: {e}"
    try:
        return await _run(prompt_key, path, override)
    finally:
        # Privacy: clipboard screenshots may contain secrets (tokens, chats,
        # credentials). Delete the temp file as soon as analysis is done.
        try:
            os.unlink(path)
        except OSError:
            pass


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if vision_client is None:
        return [
            TextContent(
                type="text",
                text="Error: GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys",
            )
        ]

    try:
        # Clipboard tools
        if name == "analyze_clipboard":
            text = await _run_clipboard("analyze", arguments.get("prompt"))
        elif name == "extract_text_from_clipboard":
            text = await _run_clipboard("extract_text")
        elif name == "describe_ui_from_clipboard":
            text = await _run_clipboard("describe_ui")
        elif name == "diagnose_error_from_clipboard":
            text = await _run_clipboard("diagnose_error")
        elif name == "code_from_clipboard":
            text = await _run_clipboard("code_from_screenshot")

        # File-path tools
        elif name == "analyze_image":
            text = await _run(
                "analyze", arguments["image_path"], arguments.get("prompt")
            )
        elif name == "extract_text":
            text = await _run("extract_text", arguments["image_path"])
        elif name == "describe_ui":
            text = await _run("describe_ui", arguments["image_path"])
        elif name == "diagnose_error":
            text = await _run("diagnose_error", arguments["image_path"])
        elif name == "understand_diagram":
            text = await _run("understand_diagram", arguments["image_path"])
        elif name == "analyze_chart":
            text = await _run("analyze_chart", arguments["image_path"])
        elif name == "code_from_screenshot":
            text = await _run("code_from_screenshot", arguments["image_path"])
        else:
            text = f"Unknown tool: {name}"

        return [TextContent(type="text", text=text)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main() -> None:
    global vision_client
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        vision_client = VisionClient(api_key)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def run() -> None:
    """Entry point for `python -m clipboard_vision_mcp` and console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
