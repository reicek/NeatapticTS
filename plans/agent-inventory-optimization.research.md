# Agent Inventory Optimization — Research Report

**Date:** 2026-08-15
**Session:** f3803217-b924-4fce-9637-a0d727e718e1
**Specialists:** 9 (6 inventory analysts + 3 Copilot standards researchers, all glm-5.2:cloud)
**Scope:** 31 agents, 66 skills, 5-tier architecture

---

## Specialist #1: TDD Loop Completeness

31 agents: 8 Tier-1, 3 Tier-2, 19 Tier-3, 1 Tier-4.

TDD Phase Coverage Matrix:

- RED (03-red-testing): Partial — phantom `planning-test-strategy-coordinator` reference in delegation table (line 323). 11 delegate agents. Strong test authoring (unit-test-writer + property-based-test-writer).
- IMPLEMENT (04-implementing): Full — 13 delegates, clean T1→T2→T3 chain, severity-gated review (TRIVIAL skips, FULL dispatches 1 of 5 POV reviewers).
- GREEN (05-green-testing): Full — 13 delegates, comprehensive validation matrix, explicit loop-back protocol, hard GPU gate.
- DOCUMENT (06-documenting): Partial — 8 delegates, good recon/review but no dedicated writer specialist. 06 does writing inline.
- LOG (07-logging): Partial — 3 delegates (thinnest phase), no session-summarizer specialist.
- Overall TDD loop completeness: 85%

Recommendations:

- R1 (0.95): Fix phantom `planning-test-strategy-coordinator` — either create the agent or remove the reference. `unit-test-writer` already creates fixtures.
- R2 (0.75): Add `session-summarizer` Tier-3 specialist under `07-logging` to collect evidence and produce structured summaries.
- R3 (0.70): Add `docs-writer` Tier-3 specialist under `06-documenting` for JSDoc/README/example writing.
- R4 (0.45): Consider merging 4 browser specialists — optional, current separation provides cleaner context isolation.
- R5 (0.85): 31 agents is near-optimal — do not reduce count. Each specialist has documented justification.

---

## Specialist #2: Agent Overlap & Consolidation

No mission-level redundancy found. All shared skills justified. All browser specialists needed (context isolation). All 8 reviewers needed (distinct POVs). All 4 scouts needed (clear boundaries). solid-split removal was correct (0.85).

Key findings:

- Consolidation candidate: `research-codebase-coordinator` → fold into `02-researching` (0.70). Single consumer, redundant scout dispatch, 02-researching already does synthesis.
- Dead references: `research-synthesis-specialist` in 02-researching line 212 (0.95), 10 phantom scouts in research-codebase-coordinator lines 101-106 (0.95)
- Orphan skills: `updating-agent-frontmatter`, `updating-skill-frontmatter` — no agent carriers (0.80)
- `implementation-executor` justified at Tier-2 (context isolation for write phase)
- All review specialists KEEP SEPARATE (0.80-0.90)
- All browser specialists KEEP SEPARATE (0.85-0.90)

Recommendations:

1. Consolidate `research-codebase-coordinator` into `02-researching` (0.70)
2. Remove phantom `research-synthesis-specialist` reference (0.95)
3. Remove 10 phantom scout names from `research-codebase-coordinator` (0.95)
4. Wire orphan skills to `agent-maintenance-coordinator` (0.80)
   5-8. Keep all specialist groups separate (0.85)

---

## Specialist #3: Missing Capability & Gap

CRITICAL: 14 phantom agents referenced but non-existent!

Phantom agents (by severity):

- CRITICAL: `helping-gap-resolution-coordinator` (referenced by ALL 7 T1 orchestrators, 36 grep matches, 0.97), `repo-cortex-scout` (0.95)
- IMPORTANT: `agent-frontmatter-auditor` (0.90), `skill-frontmatter-auditor` (0.90), `model-name-auditor` (0.90), `mcp-runtime-scout` (0.80), `code-quality-auditor` (0.85), `failure-triage-specialist` (0.85), `unit-test-runner`, `browser-runtime-scout`
- NICE-TO-HAVE: `research-synthesis-specialist` (0.70), `nge-benchmark-scout`, `worker-payload-scout`, `determinism-scout`

Orphan skills: `updating-agent-frontmatter`, `updating-skill-frontmatter` (0.95)

Domain gaps: evolution correctness, ONNX parity, WebGPU parity, worker transport, multithread evaluation (0.55-0.70)

Gates without owners: convergence-tracker, cortex-embeddings, cortex-first-search, cortex-index (phantom owner), delegate-skill-coverage, folder-quality, specialist-review (0.55-0.85)

Proposed 10 new agents:

1. `gap-resolution-coordinator` (Tier 2) — CRITICAL (0.97)
2. `repo-cortex-scout` (Tier 3) — CRITICAL (0.95)
3. `code-quality-auditor` (Tier 3) — IMPORTANT (0.85)
4. `failure-triage-specialist` (Tier 3) — IMPORTANT (0.85)
5. `frontmatter-auditor` (Tier 3) — IMPORTANT, consolidates 3 phantoms + 2 orphan skills (0.90)
6. `mcp-runtime-scout` (Tier 3) — IMPORTANT (0.80)
7. `research-synthesis-specialist` (Tier 2) — NICE-TO-HAVE (0.70)
8. `evolution-correctness-reviewer` (Tier 3) — FUTURE (0.65)
9. `onnx-parity-reviewer` (Tier 3) — FUTURE (0.60)
10. `webgpu-parity-reviewer` (Tier 3) — FUTURE (0.55)

---

## Specialist #4: Tier Architecture & Delegation

31 agents: 8 T1, 3 T2, 19 T3, 1 T4. Healthy pyramid.

- Max delegation depth: 3 hops (T1→T2→T3), median 2 — optimal (0.9)
- 04-implementing and 05-green-testing have 13 delegates each — overloaded (0.7)
- agent-maintenance-coordinator: fan-in 8 (all T1) — healthy reuse, critical hub (0.85)
- plan-scout: fan-in 7 — possibly infrastructural rather than specialist (0.7)
- implementation-executor: borderline Tier-2 (only 2 scout delegates) (0.55)
- No circular delegation risk — monotonic tier edges (0.95)

Top 3 recommendations:

1. Extract Tier-2 review coordinator for 6 review specialists shared by 04+05, cut fan-out 13→8 (0.70)
2. Re-examine plan-scout's role — may be infrastructural, not specialist (0.70)
3. Treat agent-maintenance-coordinator as protected critical hub with fallback path (0.85)

---

## Specialist #5: Skill Distribution & Routing

66 skills, routing table PASS (fresh).

- 2 orphan skills: `updating-agent-frontmatter`, `updating-skill-frontmatter` → add to `agent-maintenance-coordinator` (0.95)
- 04-implementing over-loaded: 24 skills (13 domain) → reduce to ~11 by distributing domain skills (0.70)
- `webgpu` should be `user-invocable: true` (siblings are) (0.80)
- `execute` user-invocable: possibly should be `false` (meta-orchestration skill) (0.60)
- solid-split migration correctly reflected (4 carriers: 01, 02, 04, boundary-mapper)
- No stale routing entries

Recommendations:

1. Add orphan skills to `agent-maintenance-coordinator` (0.95)
2. Reduce 04-implementing skill load 24→~11 (0.70)
3. Set `webgpu` user-invocable: true (0.80)
4. Review `execute` user-invocable setting (0.60)

---

## Specialist #6: Standards & Model Compliance

CRITICAL: ALL 31 agents use `kimi-k2.7-code:cloud` — NONE use `glm-5.2:cloud` (0.92)

- Frontmatter: PASS (0 errors, 0 warnings)
- Quality contract sections: PASS
- Structured-v1 blocks: PASS
- user-invocable: PASS (8 true = T1 orchestrators, 23 false = specialists)
- No stale agent references
- solid-split skill correctly exists (was intentionally kept, only agent removed)
- Description quality: PASS (minor validator charset issue with non-ASCII chars, source files correct)

Recommendations:

- R1 (0.92): Update all model fields. Either glm-5.2:cloud universally, or phase-appropriate per model-routing-and-budget skill
- R2 (0.80): Add model-value allowlist enforcement to validator
- R3 (0.85): Confirm solid-split skill existence — non-issue, intentionally kept
- R4 (0.70): Fix validator YAML parser charset handling

---

## Specialist #7: GitHub Copilot Agent Standards (August 2026)

Research of latest official GitHub Copilot custom agent standards.

Key findings:

**Official frontmatter fields (required + optional):**

- `description` (required), `name`, `tools`, `model`, `target`, `user-invocable`, `disable-model-invocation`, `mcp-servers`, `metadata`
- VS Code additions: `agents` (subagent allow-list), `handoffs` (with model/send/prompt/label/agent sub-fields), `argument-hint`, `hooks` (Preview), `model` can be array (priority fallback list)
- `infer` field is DEPRECATED → use `user-invocable` + `disable-model-invocation` instead
- Prompt body max 30,000 characters

**NeatapticTS-local extensions (NOT in official Copilot spec):**

- `tier`, `skills`, `triggers`, `schemas`, `expected_output`, `tool_restrictions`, `pre_action_script`, `examples`
- `structured-v1` output block
- These work because the project ships its own validators; Copilot silently ignores unrecognized fields

**Features NOT being used:**

- Hooks (Preview): 8 lifecycle events (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PreCompact, SubagentStart, SubagentStop, Stop)
- Array-valued `model` (priority fallback list)
- `target` field (env-scoping)
- `metadata` object
- `argument-hint` on agents (used on skills only)
- Plugins (package-based delivery)
- Tool search (on-demand tool loading)
- LSP servers for code intelligence
- Cloud sandbox / local sandbox
- Copilot Memory
- `/fleet` parallel tasks
- `/research` slash agent
- `/chronicle` session history
- Rubber-duck built-in agent
- Customization Evaluations (Preview)

**Potential issues found:**

- `model:` field was documented as removed (ORCHESTRATION_GUIDE.md) but is still present on all agents
- `search` and `todo` tool aliases may not resolve in CLI's agentsResolveToolAliases
- `structured-v1` enforcement only via project validators, not Copilot runtime
- Skills in frontmatter are a NeatapticTS pattern, not official Copilot binding

---

## Specialist #8: GitHub Copilot Chat Features (August 2026)

Research of Copilot Chat robust features that could benefit NeatapticTS.

**Key features available:**

1. **Handoffs**: Sequential guided workflows between agents with button-driven transitions (already partially used)
2. **Agent Skills**: Open standard (agentskills.io), `context: fork` mode, `/create-skill` command
3. **MCP Integration**: Tools, Resources, Prompts, MCP Apps (interactive UI), Sandboxing, auto-discovery
4. **Parallel Sessions**: Multiple agent sessions simultaneously, Agents Window, multiple chats per session, side chats
5. **Subagent Orchestration**: Coordinator-Worker pattern, parallel code analysis, multi-model consensus, nested up to 5 levels, recursive agents
6. **Memory Tool (Preview)**: User scope (`/memories/`), Repository scope (`/memories/repo/`), Session scope (`/memories/session/`)
7. **Copilot Memory (Preview)**: GitHub-hosted, repository-scoped, cross-agent, 28-day TTL
8. **Session Chronicle**: `/chronicle:standup`, `/chronicle:tips`, `/chronicle:cost-tips`, `/chronicle:search`
9. **Hooks (Preview)**: 8 lifecycle events, agent-scoped hooks in frontmatter, JSON stdin/stdout, can block/inject/approve
10. **Customization Evaluations (Preview)**: Analyzes customization files for contradictions, ambiguity, conflicts
11. **Thinking Effort Control**: None, Low, Medium, High — configurable reasoning depth
12. **BYOK**: Bring Your Own Key for local models (Ollama), works without Copilot plan
13. **Plan Agent**: 4-phase workflow (Discovery → Alignment → Design → Refinement)
14. **Research Agent (Preview)**: `/research <topic>` produces cited Markdown reports
15. **Autopilot Mode**: Continuous autonomous iteration, auto-approve, auto-retry, Advanced Autopilot with separate completion model

**Features that could benefit NeatapticTS:**

- Hooks: PostToolUse to run `npx tsc --noEmit` or `npm test` after edits; PreToolUse to block dangerous operations
- Memory: Store repository-level architecture decisions, preferred patterns, common pitfalls
- Customization Evaluations: Analyze agent files for quality issues
- Session Chronicle: Track work across sessions, generate standup reports
- Array-valued model: Priority fallback list for cost optimization
- MCP Resources: Expose neural network topology data as structured context

---

## Specialist #9: Model Routing & Cost Optimization (August 2026)

Research of model options and cost optimization for GitHub Copilot custom agents.

**Key findings:**

1. **`glm-5.2:cloud` is a SELF-HOSTED local Ollama model** (free, runs at 127.0.0.1:11434), NOT a GitHub Copilot cloud model
2. **`kimi-k2.7-code:cloud` IS an official GA Copilot model** — cheapest cloud model with coding focus ($0.95 input / $4.00 output per 1M tokens)
3. **Validator strict allowed models**: `glm-5.2:cloud (ollama)`, `kimi-k2.7-code:cloud`. NOTE: `anthropic/claude-sonnet-4-20250514` is currently in the validator but is NOT approved — must be removed per user directive.

**"Claude Haiku 4.6" does NOT exist** — only Haiku 4.5 is GA. The model-routing-and-budget skill references "Claude Haiku 4.6" which should be corrected.

**NOTE: Neither Claude Sonnet nor Claude Haiku are approved models for NeatapticTS.** Only `glm-5.2:cloud` and `kimi-k2.7-code:cloud` are approved. The `anthropic/claude-sonnet-4-20250514` entry in the validator must be removed. The model-routing-and-budget skill must be updated to remove all Claude references and use only the two approved models.

**Claude Sonnet 5** ($2.00/$10.00) is cheaper than Sonnet 4.6 ($3.00/$15.00) and is GA — may be worth considering **in the future** if explicitly approved by the user. Not currently approved.

**Recommended model assignment matrix:**

| Agent Type                    | Recommended Model             | Justification                                |
| ----------------------------- | ----------------------------- | -------------------------------------------- |
| Heavy: Implementation/Editing | `glm-5.2:cloud` (local, free) | Deep reasoning, edge-case handling. Free.    |
| Heavy: Planning/Architecture  | `glm-5.2:cloud` (local, free) | Broad reasoning needed. Free.                |
| Heavy: Red Testing            | `glm-5.2:cloud` (local, free) | Test contracts need careful judgment. Free.  |
| Mid: Green Testing            | `kimi-k2.7-code:cloud`        | Verification mostly mechanical. Cheap cloud. |
| Mid: Research/Exploration     | `kimi-k2.7-code:cloud`        | Retrieval/summarization. Cheap, fast.        |
| Mid: Code Review/Security     | `kimi-k2.7-code:cloud`        | Read-only review. Cheap, fast.               |
| Mid: Documentation            | `kimi-k2.7-code:cloud`        | Writing tasks. Cheap.                        |
| Light: Logging/Summarization  | `kimi-k2.7-code:cloud`        | Most lightweight tasks. Cheapest cloud.      |
| Light: Scouts (all)           | `kimi-k2.7-code:cloud`        | Read-only recon. Cheap, parallel-friendly.   |
| Light: Browser specialists    | `kimi-k2.7-code:cloud`        | Narrow scoped. Cheap.                        |
| Light: Learning events        | `kimi-k2.7-code:cloud`        | One-shot auxiliary. Cheapest.                |

**Cost analysis:**

- glm-5.2:cloud = $0 (local Ollama, limited by hardware)
- kimi-k2.7-code:cloud = ~$0.0175 per typical agent interaction (10K input + 2K output tokens)
- Using tiered strategy saves significant costs vs all-glm-5.2 (which would be free but hardware-limited) or all-cloud (which would be expensive)
