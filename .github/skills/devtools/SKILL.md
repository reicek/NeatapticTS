---
name: devtools
description: 'Use when: an agent or gate requires the devtools skill name. Delegates durable Chrome DevTools MCP policy to chrome-devtools-mcp.'
argument-hint: 'Describe the browser testing, performance measurement, or UI validation task.'
user-invocable: false
disable-model-invocation: false
---

# DevTools Alias

This skill is a thin registered alias that maps the `devtools` name to the canonical
[chrome-devtools-mcp](../chrome-devtools-mcp/SKILL.md) skill. All durable Chrome DevTools MCP
policy lives in that skill.

## Visible Browser Note

For real GPU/performance validation, the browser window must be visible. The
`--headless=false` flag is not reliable in all environments; the canonical
workaround is to launch Chrome with `--remote-debugging-port=9222` and connect
the DevTools MCP to that existing instance. See the **Visible Browser Window**
section in [chrome-devtools-mcp](../chrome-devtools-mcp/SKILL.md) for the
PowerShell command and the programmatic `launchVisibleChrome(url)` helper.
