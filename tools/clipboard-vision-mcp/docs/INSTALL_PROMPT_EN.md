# AI install prompt (English)

Copy-paste this into any AI coding assistant (DeepSeek, GLM, Claude, GPT, Gemini, ...) and it will set up `clipboard-vision-mcp` for you end-to-end.

---

```
You are going to install and configure the `clipboard-vision-mcp` MCP server on my machine so that I can paste screenshots in my MCP-capable coding client (Opencode, Claude Code, Cursor, Cline, Continue, ...) and have text-only models see them via Groq + Llama-4 Scout.

Repository: https://github.com/Capetlevrai/clipboard-vision-mcp

Please:

1. Detect my operating system (Windows / macOS / Linux) and the Linux display server (X11 vs Wayland) if relevant.
2. Check that Python 3.10+ and git are installed. If not, tell me the exact command to install them for my OS and stop.
3. Clone the repo into a sensible location (e.g. `~/tools/clipboard-vision-mcp` on Unix, `%USERPROFILE%\tools\clipboard-vision-mcp` on Windows) and run `pip install -e .`.
4. Install the OS-specific clipboard dependency:
   - Windows: nothing extra.
   - macOS: `brew install pngpaste` (optional but recommended).
   - Linux + Wayland: `sudo apt install wl-clipboard` (or distro equivalent).
   - Linux + X11: `sudo apt install xclip` (or distro equivalent).
5. Ask me for my Groq API key (free at https://console.groq.com/keys). Do NOT hardcode it in any committed file — only write it into my local MCP client config.
6. Detect which MCP client I'm using (ask if uncertain) and write the correct config:
   - Opencode: add an `mcp.clipboard-vision` entry to `opencode.json` with `"command": ["python", "-m", "clipboard_vision_mcp"]` and the `GROQ_API_KEY` env var. KEYBINDS GOTCHAS (critical, do not skip): (a) opencode.json rejects unknown top-level keys — NEVER put a `keybinds` block in opencode.json or opencode will fail to start; keybinds go in a SEPARATE `tui.json` next to opencode.json. (b) There is NO `input_paste_image` action in opencode; paste (text AND images) is the single `input_paste` action, which opencode's TUI reads from the OS clipboard. Bind `input_paste` to BOTH `ctrl+v` and `alt+v` in `tui.json` so Alt+V pastes an image (see examples/tui.json). (c) After writing both files, validate each as JSON before finishing.
   - Claude Code / Cursor / Cline / Continue: follow the shape in docs/CLIENTS.md from the repo.
7. Verify the server starts: run `python -m clipboard_vision_mcp` for ~2 seconds and confirm it does not crash (it should stay idle on stdin).
8. Tell me exactly how to test it: copy a screenshot, open my client, ask "use analyze_clipboard and describe what I just copied".

Constraints:
- Never commit or echo my API key in plaintext to any shared location.
- If any step fails, stop and show me the full error + a proposed fix, don't silently continue.
- Prefer editing existing config files over replacing them — merge JSON, don't overwrite.
- Report back at the end with: install path, edited files, test command.
```

---

**Tip.** Paste this *as-is*. Any decent coding assistant with file-system and shell tools will handle it. If the assistant can't run shell commands, it will print the exact commands for you to run.
