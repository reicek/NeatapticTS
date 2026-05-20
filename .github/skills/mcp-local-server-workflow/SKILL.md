---
name: mcp-local-server-workflow
description: 'Design and validate local MCP runtime-visibility servers for NeatapticTS. Use when planning or implementing repo-static workflow facts, direct-MCP validation gates, bridge-required runtime evidence, manual-only client-state boundaries, MCP tool/resource schemas, or trust-boundary rules.'
argument-hint: 'Name the runtime facts, whether they are repo-static, direct-MCP, bridge-required, or manual until a documented API exists, the planned MCP component, bridge uncertainty, and required validation gates.'
user-invocable: false
disable-model-invocation: false
---

# MCP Local Server Workflow

Use this skill when work needs a local MCP server, VS Code bridge, or
MCP-backed validation path for NeatapticTS agentic workflow facts.

## Workflow

1. Separate facts into four buckets: repo-static, direct-MCP,
   bridge-required, and manual until a documented API exists.
2. Prefer a local Node/TypeScript `stdio` MCP server for repo-static facts such
   as the active workflow snapshot and deterministic customization inventory.
3. Treat bridge-required facts as future extension-host or Copilot client
   observations that the shipped direct-MCP entrypoints cannot read today, and
   require source, freshness, and client context before those facts can be
   presented as runtime evidence.
4. Expose read-only MCP tools/resources by default and mark read-only behavior
   with MCP annotations where the SDK supports them.
5. Keep validation MCP tools allow-listed by the active step packet. They may
   expose allow-list introspection and run only exact commands named in the
   active step packet, reporting command, exit code, and concise output without
   using a shell.
6. Treat hooks as deterministic lifecycle automation or audit context, not as a
   substitute for runtime fact APIs or a shortcut around the bridge-required
   versus manual-only boundary.
7. Update `plans/Agentic_Workflow_Architecture.plans.md` with capability
   evidence, bridge blockers, validation results, and the next phase packet.

## Runtime Fact Classification

Use this quick classification pass before assigning a fact to an MCP component.

- `repo-static`: the fact comes from repository files or deterministic scripts
   and does not depend on current client state. Example: the active phase/step
   packet or the customization inventory counts.
- `direct-MCP`: the fact can be produced by the shipped MCP entrypoints from
   the active step packet and exact allow-listed commands without using a shell.
   Example: the current validation allow-list or the exit code for an allowed
   validation command.
- `bridge-required`: the fact depends on extension-host or Copilot client
   observations that the shipped direct-MCP surface cannot read today. Example:
   a model snapshot or a hook-derived observation. Record source, freshness, and
   client context before treating it as runtime evidence.
- `manual until a documented API exists`: the fact is visible to a human in the
   current client but the repo does not ship a supported API to read it.
   Examples: the selected active agent, the full live agent list, tool-picker
   state, and the current selected model UI state.

## Component Boundaries

- `neataptic-workflow-mcp`: repo-static workflow snapshot plus deterministic
   customization inventory from files and existing deterministic scripts.
- `neataptic-validation-mcp`: active-step allow-list introspection plus
   shell-free exact-command execution for validation gates named by the current
   step packet.
- `neataptic-vscode-bridge`: provisional future owner for bridge-required model
   snapshots or hook observations only; it is not part of the shipped direct-MCP
   surface today.
- Manual until a documented API exists: selected active agent, the full live
   agent list, tool-picker state, and the current selected model UI state stay
   outside the shipped MCP surfaces.

## Guardrails

- Do not use MCP as a general shell.
- Do not store live client observations without source, timestamp, and client
  context.
- Do not imply a checked-in MCP workspace configuration, a plugin-packaged
   bridge, or direct access to manual-only client facts when the repo does not
   ship those surfaces.
- Do not tune agent tags, skill descriptions, model routing, or trigger evals
  until runtime visibility is scoped.
- Do not edit generated docs to record MCP progress; use the active tracker.

## Sources

- VS Code MCP developer guide: `https://code.visualstudio.com/api/extension-guides/ai/mcp`
- VS Code Copilot MCP server customization guide: `https://code.visualstudio.com/docs/copilot/customization/mcp-servers`
- VS Code AI extensibility overview: `https://code.visualstudio.com/api/extension-guides/ai/ai-extensibility-overview`
- VS Code Language Model API guide: `https://code.visualstudio.com/api/extension-guides/language-model`
- VS Code agent plugins documentation: `https://code.visualstudio.com/docs/copilot/customization/agent-plugins`
- VS Code hooks documentation: `https://code.visualstudio.com/docs/copilot/customization/hooks`

Keep claims about MCP runtime behavior bounded to the shipped NeatapticTS
surfaces above. If a later note needs protocol-wide claims, add a separate
protocol source before making them.