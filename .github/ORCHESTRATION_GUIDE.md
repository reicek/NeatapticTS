# Orchestration Guide

> **Canonical reference:** `.github/skills/execute/SKILL.md` is the single source of truth for delegation, dispatch, and the RED→IMPLEMENT→GREEN loop. This guide is a quick-reference cheat-sheet only.

## Tier Graph
| Tier | Role | Can dispatch to |
|------|------|-----------------|
| 0 | Root orchestrator (Agent Zero) | Tier 1 only |
| 1 | SDLC orchestrators (01-07) | Tier 2, 3, 4 |
| 2 | Coordinators | Tier 3, 4 |
| 3 | Scouts & specialists | Tier 4 (rare) |
| 4 | One-shot auxiliaries | — |

Delegation is strictly downward. No T1→T1. Agent Zero manages the main loop; T1 agents manage sub-loops within their phase.

## Dispatch Protocol
1. Call `neataptic-dispatch-mcp / build_dispatch_packet` before every dispatch.
2. Use the returned `dispatch_packet` with the `task` tool.
3. MCP tool names use HYPHENS: `neataptic-dispatch-mcp-build_dispatch_packet`.
4. Prompt limits: trivial=200 chars, moderate=500, complex=1000.

## RAG-Based Dispatch
- Dispatch with only a step/slice ID + "Load context via Cortex MCP / get_slice_context."
- NEVER embed inline instructions, file lists, or design context.
- If the plan lacks context, dispatch 01-planning to update it first.

## RED → IMPLEMENT → GREEN Loop
0. Plan Verification Gate (fresh 01-planning, unless bypassed by pragmatic mode)
1. Red Testing (03-red-testing) — complex slices only
2. Implementation (04-implementing) → Shared Validation Gate → Specialist Review (severity-gated)
3. Green Testing (05-green-testing)
4. Loop-back: fix packet → NEW 04 → NEW 05 (convergence tracker: >4 iterations → escalate)
5. Advance: update plan, next step — then Phase Compression (dispatch 07-logging)

## Time-Boxing
- Soft cap: ~20 min (trivial/moderate), ~30 min (complex).
- If an agent exceeds cap+50%, check progress; if static, stop and re-dispatch. The orchestrator MAY stop a running agent and pass follow-up instructions via write_agent.

## Concurrency
| Variable | Default | Purpose |
|----------|---------|---------|
| `COPILOT_SUBAGENT_MAX_CONCURRENT` | 2 (BYOK) / 10 (plan-tier) | Max concurrent sub-agents |
| `COPILOT_SUBAGENT_MAX_DEPTH` | 6 | Max delegation chain depth |

## Model Resolution Note

`model:` is a **retained local extension** — it is not part of the standard
Copilot agent frontmatter spec, but NeatapticTS keeps it on all agents for
per-agent model routing. The CLI honors it for local dispatch; per-dispatch
override is available via the `task` tool's `model` parameter. Agent `model:`
frontmatter fields use `glm-5.3-flash:cloud` (local Ollama, free) and
`kimi-k2.7-code:cloud` (Copilot cloud). The CLI may append `(ollama)` suffix
for local models. If dispatch returns 400 errors, verify the model name is
valid. See `agent-model-reference.md`.

## Tool Aliases
Some aliases may not resolve in all CLI versions. Use explicit names: `view` (not `read`), `grep` (not `search`), `powershell` (not `execute`), `task` (not `agent`), SQL `todos` (not `todo`).
