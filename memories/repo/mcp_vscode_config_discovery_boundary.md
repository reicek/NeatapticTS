# MCP Config Discovery Boundary

## Problem

`.vscode/` is gitignored (`.gitignore` line 36: `/.vscode/`).

`file_search` and `grep_search` (without `includeIgnoredFiles: true`) will **not** find
`.vscode/mcp.json`. Agents that rely on `file_search` to verify the MCP binding will receive a
false negative and incorrectly conclude the file does not exist.

## Correct discovery path

Use `list_dir` + `read_file` instead:

```
list_dir("c:\\NeatapticTS\\.vscode")       → confirms mcp.json is present
read_file("c:\\NeatapticTS\\.vscode\\mcp.json", 1, 40)
```

Or from a terminal:

```powershell
Get-Content "c:\NeatapticTS\.vscode\mcp.json"
```

## Current MCP server binding (as of 2026-05-22)

File: `c:\NeatapticTS\.vscode\mcp.json`

| Server | Args |
|--------|------|
| `neataptic-workflow-mcp` | `--plan=plans/mcp-active-binding.plans.md` |
| `neataptic-validation-mcp` | `--plan=plans/mcp-active-binding.plans.md` |
| `neataptic-gate-mcp` | _(no --plan arg)_ |

Both workflow and validation servers bind to `plans/mcp-active-binding.plans.md`, which is a
**permanent [WIP]** file — never archived.

## Guidance for orchestrators

- Do not infer "file not found" from a negative `file_search` result for anything under `.vscode/`.
- When a research step must verify the MCP `--plan` binding, use `list_dir` + `read_file` as the
  primary path; `file_search` is unreliable for gitignored directories.
