# Orchestration Guide

> **Canonical reference:** `.github/skills/execute/SKILL.md` is the single source of truth for delegation, dispatch, and the RED→IMPLEMENT→GREEN loop. This guide is a quick-reference cheat-sheet only.

> **Git rule source of truth:** `.github/copilot-instructions.md` — mirrored verbatim below; binding on every agent in every tier.

## CRITICAL RULE — GIT: READ-ONLY INSPECTION ALLOWED, MUTATION BANNED

**NEVER run a mutating git command.** Read-only inspection is ALWAYS allowed: `git status [-v]`, `git diff`, `git log`, `git show`, `git blame`, `git remote -v`, `git rev-parse`, `git ls-files`; `--no-pager` is permitted (e.g. `git --no-pager log`). All mutating commands stay BANNED: `git add`, `git commit`, `git push`, `git pull`, `git fetch`, `git checkout`, `git reset`, `git restore`, `git revert`, `git stash`, `git clean`, `git merge`, `git rebase`, `git apply`, and any other command that mutates the worktree, index, refs, or remote. Mutating git is UNINSTALLED here — running it has destroyed hours of work by reverting files. All file changes must use the `edit` or `create` tools ONLY. If you need to see file contents, use the `view` tool. Read-only inspection is EVIDENCE ONLY and never a substitute for the write path. This rule applies to ALL agents and ALL operations.

## Tier Graph

| Tier | Role                           | Can dispatch to |
| ---- | ------------------------------ | --------------- |
| 0    | Root orchestrator (Agent Zero) | Tier 1 only     |
| 1    | SDLC orchestrators (01-07)     | Tier 2, 3, 4    |
| 2    | Coordinators                   | Tier 3, 4       |
| 3    | Scouts & specialists           | Tier 4 (rare)   |
| 4    | One-shot auxiliaries           | —               |

Delegation is strictly downward. No T1→T1. Agent Zero manages the main loop; T1 agents manage sub-loops within their phase.

## Dispatch Protocol

1. Call `neataptic-dispatch-mcp / build_dispatch_packet` before every dispatch.
2. Use the returned `dispatch_packet` with the `task` tool.
3. MCP tool names use HYPHENS: `neataptic-dispatch-mcp-build_dispatch_packet`.
4. Prompt limits: trivial=200 chars, moderate=500, complex=500.

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

## Orchestration Doctrine

Five binding rules for every dispatch. Full text: `execute` skill §9.0.

1. **Thin scope** — one objective, minimum file set; no two live dispatches own the same file.
2. **Parallel scouts** — independent read-only scouts run in parallel, not sequentially.
3. **Central merge** — findings merge at one point: the orchestrator. No agent-to-agent chaining.
4. **Parallel executors** — parallel writes allowed only over non-overlapping file sets; overlap is serialized.
5. **Time-budget exit** — an agent over budget stops and returns a gap report for a fresh takeover.

## Binding Orchestration Mandates

Binding on every dispatch. Full text: `execute` skill §9.0.1 (RPE-6/7/8) and `.github/copilot-instructions.md` → "Binding Orchestration Mandates".

- **Never reuse or recycle an agent** — one instance owns one assignment; a finished, failed, or stalled instance is never resumed for new work.
- **Every dispatch is a FRESH instance** on a small, targeted slice (one objective, one boundary, minimum file set).
- **RAG by slice ID only** — the prompt carries the slice/step ID plus a RAG-load instruction; no inlined file lists, code, or design context.
- **Sub-agents own their work, never forward** — no sideways re-dispatch or chaining; escalate blockers up to the orchestrator.
- **Stalled or narration-stalled = STOP, never nudge** — stop it, capture the gap report, dispatch a fresh instance seeded with it.
- **Bounded fan-out** — reserve at least one orchestrator slot; size the fan-out to the task, not to free slots.

`write_agent` is permitted only for a small clarification inside an agent's own assignment scope — never to recycle an agent, hand off unfinished work, or substitute for a fresh instance.

## Time-Boxing

- Soft cap: ~20 min (trivial/moderate), ~30 min (complex).
- If an agent exceeds cap+50%, check progress; if static, it is STALLED — stop it, capture its gap report, and re-dispatch a FRESH instance seeded with that report.
- Never nudge a stalled or narration-stalled instance ("continue", "any progress?"); a nudge is a violation, not a recovery — stop and re-dispatch fresh.

## Concurrency

| Variable                          | Default                   | Purpose                    |
| --------------------------------- | ------------------------- | -------------------------- |
| `COPILOT_SUBAGENT_MAX_CONCURRENT` | 2 (BYOK) / 10 (plan-tier) | Max concurrent sub-agents  |
| `COPILOT_SUBAGENT_MAX_DEPTH`      | 6                         | Max delegation chain depth |

Always reserve ≥ 1 slot for the orchestrator.

## Model Resolution Note

`model:` is a **retained local extension** — it is not part of the standard
Copilot agent frontmatter spec, but NeatapticTS keeps it on all agents for
per-agent model routing. The CLI honors it for local dispatch; per-dispatch
override is available via the `task` tool's `model` parameter. Agent `model:`
frontmatter fields use `minimax-m3.1-flash:cloud` (local Ollama, free) and
`minimax-m3.1-flash:cloud` (Copilot cloud). The CLI may append `(ollama)` suffix
for local models. If dispatch returns 400 errors, verify the model name is
valid. See `agent-model-reference.md`.

## Tool Aliases

Some aliases may not resolve in all CLI versions. Use explicit names: `view` (not `read`), `grep` (not `search`), `powershell` (not `execute`), `task` (not `agent`), SQL `todos` (not `todo`).
