# MCP client configuration

All clients below speak MCP stdio. The command is always:

```
python -m clipboard_vision_mcp
```

with `GROQ_API_KEY` in the environment.

## Claude Code

Edit `~/.claude/settings.json` (or `%USERPROFILE%\.claude\settings.json`):

```json
{
  "mcpServers": {
    "clipboard-vision": {
      "command": "python",
      "args": ["-m", "clipboard_vision_mcp"],
      "env": { "GROQ_API_KEY": "gsk_..." }
    }
  }
}
```

Or CLI:

```bash
claude mcp add clipboard-vision -- python -m clipboard_vision_mcp
```

## Cursor

Settings → Features → MCP → Add new:

```json
{
  "clipboard-vision": {
    "command": "python",
    "args": ["-m", "clipboard_vision_mcp"],
    "env": { "GROQ_API_KEY": "gsk_..." }
  }
}
```

## Cline / Continue

Both clients use the same `mcpServers` shape — drop it into their MCP config file.

## Generic stdio

If your client accepts raw commands, pass:

```
env GROQ_API_KEY=gsk_... python -m clipboard_vision_mcp
```
