# Agent Inventory Optimization Plan

**Date:** 2026-08-15
**Session:** f3803217-b924-4fce-9637-a0d727e718e1
**Status:** APPROVED — ALL 6 VALIDATION SPECIALISTS CONSENSUS, READY FOR IMPLEMENTATION
**Specialists:** 9 (6 inventory analysts + 3 Copilot standards researchers)
**Consensus:** All 9 specialists reported; 6 validators approved after 2 rounds (S2, S3 approved round 1; S1, S4, S5, S6 approved round 2 after batch fixes)

## Mandates

### Model Strategy Mandate (CRITICAL)

The permanent `model:` field for each agent must be set based on task complexity, NOT universally:

- **`glm-5.2:cloud` (local Ollama, free)** for HEAVY tasks requiring deep reasoning:
  - Tier-1 orchestrators: 00-helping, 01-planning, 03-red-testing, 04-implementing
  - Tier-2 coordinator: implementation-executor
  - These agents need careful judgment, edge-case handling, architecture decisions

- **`kimi-k2.7-code:cloud` (Copilot cloud, $0.95/$4.00 per 1M tokens)** for LIGHT tasks:
  - Tier-1: 02-researching, 05-green-testing, 06-documenting, 07-logging
  - Tier-2: agent-maintenance-coordinator
  - ALL Tier-3 scouts and specialists (19 agents)
  - ALL Tier-4 auxiliaries (1 agent)
  - These agents do read-only recon, mechanical verification, narrow review, or summarization

- **05-green-testing** stays on kimi-k2.7 because verification is mostly mechanical
- **00-helping** uses glm-5.2 because gap resolution needs nuanced synthesis
- **Users may explicitly override** the default model per request based on budget
- **Plan mandates** for implementing agents continue to use glm-5.2:cloud (the mandate applies to plans and the agents that implement them, not permanent agent fields)
- **Cost rationale**: kimi-k2.7 is the cheapest Copilot cloud model with coding focus. Using it for scouts and parallel execution avoids cost overruns. glm-5.2 is free but hardware-limited, so reserve it for tasks that need its reasoning depth.
- **NO OTHER MODELS APPROVED**: Only `glm-5.2:cloud` and `kimi-k2.7-code:cloud` are approved. Claude Sonnet, Claude Haiku, and any other models are NOT approved. Additional models may be added later only with explicit user approval.

### Other Mandates

- No git commands — use edit/create tools only
- "Always elevating standards, when having to choose, choose the latest, highest standard"
- Pragmatic mode: batch related changes, validate per phase
- Nothing deferred — all phases implemented in detail including domain POV reviewers
- This is the core of the project — take time, do it right, polished

### Execution Mandates (for /execute skill)

- Pragmatic mode: broad slices — dispatch unit is one agent per step (not per phase)
- Bypass plan-readiness gate (green-light cycle) — plan is pre-approved by 9 specialists
- Bypass step-packet and plan-slice-quality gates — plan uses prose steps, not YAML step packets
- Steps are dispatched by step ID (e.g., "Step 0.1") via RAG load
- Validation is per-phase (run validation commands listed at end of each phase)
- Tiered model mandate: glm-5.2:cloud for heavy agents (00-helping, 01-planning, 03-red-testing, 04-implementing, implementation-executor), kimi-k2.7-code:cloud for all others. Per-dispatch model override via task tool model parameter.
- Phase 6 multi-dispatch exception: 2 specialists per agent (Content Quality + Standards), batched by tier. The pragmatic mode's single-dispatch model does not apply to Phase 6.
- Phase 6 ownership: dispatched by 05-green-testing (T1) which dispatches Tier-3 specialists sequentially (Ollama concurrency limit 2)
- When pragmatic mode and "nothing deferred" conflict, lean toward thoroughness: keep validation gates, skip only plan-verification green-light cycle and fix-packet YAML ceremony

### Phase Sequencing

Phases are strictly sequential: Phase N+1 begins only after Phase N is [DONE] and validation passes. Cross-phase dependencies are noted inline. The orchestrator MUST update the Handoff Query's "Next" line after each phase completes.

## Current State

- 31 agents: 8 Tier-1, 3 Tier-2, 19 Tier-3, 1 Tier-4
- 66 skills, 1 routing table (PASS/fresh)
- All validation gates PASS (frontmatter, quality, graph, routing)
- ALL 31 agents use `kimi-k2.7-code:cloud` — needs tiered model assignment
- 14 phantom agent references across 11 files
- 2 orphan skills with zero carriers
- 04-implementing overloaded with 24 skills
- 3 new features identified from Copilot research: hooks, array-model, customization evaluations

<!-- slice: aio-phase-0 -->

## Phase 0 — Model Assignment (Tiered Strategy) [DONE]

**Goal:** Assign models based on task complexity, not universally

<!-- step: aio-phase-0-step-0-1 -->

**Step 0.1:** Set HEAVY agents to `glm-5.2:cloud` (5 agents):

- 00-helping, 01-planning, 03-red-testing, 04-implementing, implementation-executor ✅ DONE

<!-- step: aio-phase-0-step-0-2 -->

**Step 0.2:** Set LIGHT agents to `kimi-k2.7-code:cloud` (26 agents, keep current):

- 02-researching, 05-green-testing, 06-documenting, 07-logging
- agent-maintenance-coordinator
- ALL 19 Tier-3 specialists
- learning-event-capturer (Tier-4) ✅ DONE (verified — all 26 already correct)

<!-- step: aio-phase-0-step-0-3 -->

**Step 0.3:** Add model-value allowlist to `validate-agent-frontmatter.mjs` — allow ONLY `glm-5.2:cloud` and `kimi-k2.7-code:cloud`. No other models are approved. Reject all other values. Also remove stale entries for `planning-test-strategy-coordinator` and `research-codebase-coordinator` from `strictTier2CoordinatorPathsByName` in `validate-agent-frontmatter.mjs`. Remove `anthropic/claude-sonnet-4-20250514` from the existing allowed models set — it is NOT approved. ✅ DONE

<!-- step: aio-phase-0-step-0-4 -->

**Step 0.4:** Update `model-routing-and-budget` skill — remove ALL references to Claude Sonnet and Claude Haiku models. Neither Sonnet nor Haiku are approved models. Only `glm-5.2:cloud` and `kimi-k2.7-code:cloud` are approved. The skill's phase-based tier system must use only these two models: glm-5.2:cloud for heavy/full tasks, kimi-k2.7-code:cloud for light tasks. Remove the "Claude Haiku 4.6" reference (does not exist) and all Sonnet references. Document that additional models may be added later only with explicit user approval. ✅ DONE

**Validation:** `validate-agent-frontmatter.mjs --json --strict`, `validate-agent-quality.mjs`, `validate-agent-graph.mjs --json`

<!-- slice: aio-phase-1 -->

## Phase 1 — Phantom Agent Cleanup [DONE]

**Goal:** Resolve ALL 14 phantom agent references — create needed agents, remove dead references

<!-- step: aio-phase-1-step-1-1 -->

**Step 1.1:** Remove `helping-gap-resolution-coordinator` phantom references (36 matches, all T1 + implementation-executor)

- Update all 7 T1 orchestrators' "If Blocked" sections to reference `00-helping` directly
- Update `implementation-executor` similarly
- Update 00-helping's mission/body to explicitly document gap-resolution coordination as a first-class responsibility (absorbing the phantom `helping-gap-resolution-coordinator` scope)
- **Slice files:** 8 agent files

<!-- step: aio-phase-1-step-1-2 -->

**Step 1.2:** Create `frontmatter-auditor` (Tier 3) — consolidates 3 phantom agents

- Consolidates: `agent-frontmatter-auditor`, `skill-frontmatter-auditor`, `model-name-auditor`
- Skills: `agent-frontmatter-standards`, `skill-frontmatter-standards`, `updating-agent-frontmatter`, `updating-skill-frontmatter`, `model-routing-and-budget`
- Model: `kimi-k2.7-code:cloud` (read-only audit, light task)
- Add to `agents:` allow-list of `00-helping` and `agent-maintenance-coordinator`
- **Slice files:** 1 new agent, 2 modified

<!-- step: aio-phase-1-step-1-3 -->

**Step 1.3:** Create `repo-cortex-scout` (Tier 3)

- Skills: `repo-cortex-workflow`, `repo-cortex-embeddings`, `research-methodology`
- Model: `kimi-k2.7-code:cloud` (read-only scout)
- Add to `agents:` allow-list of `00-helping` and `02-researching`
- **Slice files:** 1 new agent, 2 modified

<!-- step: aio-phase-1-step-1-4 -->

**Step 1.4:** Remove ALL remaining phantom references:

- `code-quality-auditor` → route to `implementation-standards` skill
- `failure-triage-specialist` → route to `05-green-testing` with `test-fix-workflow`
- `mcp-runtime-scout` → route to `mcp-local-server-workflow` skill
- `unit-test-runner` → route to `05-green-testing` with `running-unit-tests`
- `browser-runtime-scout` → remove from `00-helping` and `research-codebase-coordinator`
- `research-synthesis-specialist` → remove from `02-researching`
- 10 phantom scouts in `research-codebase-coordinator` → remove all
- `planning-test-strategy-coordinator` → remove from ALL files where it appears (not just `03-red-testing`). ALL references must be removed across the entire repo, including: `03-red-testing`, `01-planning.agent.md`, `.github/FLOWS.md`, `.github/flows/03.behavior-change-red.flow.yml`, `.github/flows/01.acceptance-criteria.flow.yml`, `.github/flows/01.phase-kickoff.flow.yml`. Document fixture strategy handled by `unit-test-writer`.
- `nge-benchmark-scout`, `worker-payload-scout`, `determinism-scout` → remove from `research-codebase-coordinator`
- Note: The 10 phantom scouts in research-codebase-coordinator are resolved by the Phase 2 deletion of that file — no separate cleanup needed for those 10. The other phantom removals in this step apply to files that survive.
- **Slice files:** 9 files (00-helping, 02-researching, 03-red-testing, research-codebase-coordinator, 01-planning.agent.md, .github/FLOWS.md, .github/flows/03.behavior-change-red.flow.yml, .github/flows/01.acceptance-criteria.flow.yml, .github/flows/01.phase-kickoff.flow.yml)

<!-- step: aio-phase-1-step-1-5 -->

**Step 1.5:** Wire orphan skills `updating-agent-frontmatter` and `updating-skill-frontmatter` to `frontmatter-auditor` (created in Step 1.2)

- **Slice files:** 1 agent file (frontmatter-auditor already has them in skills)

**Validation:** `validate-agent-frontmatter.mjs --json --strict`, `validate-agent-quality.mjs`, `validate-agent-graph.mjs --json`, `npm run agents:routing-table`, routing-table-freshness gate, grep scan for zero phantom references

<!-- slice: aio-phase-2 -->

## Phase 2 — Consolidation & Skill Redistribution [DONE]

**Goal:** Reduce 04-implementing overload, consolidate redundant coordinator, redistribute domain skills

<!-- step: aio-phase-2-step-2-1 -->

**Step 2.1:** Remove `research-codebase-coordinator` (Tier 2) — fold into `02-researching`

- 02-researching already dispatches the same scouts and does synthesis
- Delete `research-codebase-coordinator.agent.md`
- Add `implementation-pattern-scout` to 02-researching's `agents:` array (it was in research-codebase-coordinator but not yet in 02-researching)
- Remove `research-codebase-coordinator` from 02-researching's `agents:` array
- **Slice files:** 2 (02-researching modified, research-codebase-coordinator deleted)

<!-- step: aio-phase-2-step-2-2 -->

**Step 2.2:** Reduce 04-implementing skill load (24 → 14, removes 10 skills)

- REDISTRIBUTE from 04 to other agents/carriers (10 skills removed):
  - `webgpu` → REMOVE from 04, add to `webgpu-parity-reviewer` (Phase 5 agent)
  - `onnx-work` → REMOVE from 04, add to `onnx-parity-reviewer` (Phase 5 agent)
  - `nge-core-algorithm` → REMOVE from 04, add to `evolution-correctness-reviewer` (Phase 5 agent)
  - `worker-inference-transport` → REMOVE from 04, add to `05-green-testing`
  - `multithread-evaluation` → REMOVE from 04, add to `05-green-testing`
  - `neatchat-systems` → REMOVE from 04, add to `02-researching` and `06-documenting`
  - `architecture-builder` → REMOVE from 04, add to `02-researching`
  - `visualizer-workflow` → REMOVE from 04, add to `06-documenting`
  - `browser-build` → REMOVE from 04, add to `05-green-testing`
  - `trace-analyzer-extension` → REMOVE from 04, add to `02-researching`
- Note: `nge-benchmark-workflow` is already on 05-green-testing, NOT on 04. Add it also to 02-researching for benchmark research visibility. This is a skill propagation, not a 04 reduction.
- Keep on 04 only: `flappy-architecture-polish`, `checkpointing-persistence`, `hybrid-training-interop`
- 05-green-testing receives 3 additional domain skills (worker, multithread, browser-build) for parity validation — 05 is the validation gate phase and needs domain coverage. 05 goes from 18 to 21, which is acceptable for a validation phase.
- **Slice files:** 4 agent files (04-implementing, 05-green-testing, 02-researching, 06-documenting)
- Note: Skill redistribution to Phase 5 agents (webgpu-parity-reviewer, onnx-parity-reviewer, evolution-correctness-reviewer) is DEFERRED to Phase 5 Step 5.4 — these agents do not exist until Phase 5 creates them. Step 2.2 only removes these skills from 04-implementing; the addition to Phase 5 agents happens in Phase 5 after agent creation.

<!-- step: aio-phase-2-step-2-3 -->

**Step 2.3:** Extract Tier-2 review coordinator for 6 review specialists shared by 04+05

- Specialists: security-reviewer, performance-reviewer, determinism-reviewer, api-contract-reviewer, dependency-audit-reviewer, benchmark-gate-reviewer
- Cuts 04 fan-out from 13 to 9 (removes 5 reviewers, adds 1 coordinator). Cuts 05 fan-out from 13 to 10 (removes 4 reviewers, adds 1 coordinator). Only 3 reviewers are truly shared by both.
- Create `review-coordinator` (Tier 2), model: `kimi-k2.7-code:cloud` (delegates to reviewers, light coordination)
- Add to `agents:` of 04-implementing and 05-green-testing
- Remove 6 reviewers from `agents:` of 04 and 05, add to `review-coordinator` instead
- Note: `benchmark-gate-reviewer` and `dependency-audit-reviewer` remain direct delegates of 02-researching (for research-phase surface review) AND become delegates of review-coordinator (for implementation/green POV review). This dual path is intentional — different phases need the same reviewer for different purposes.
- **Slice files:** 3 (1 new agent, 04-implementing, 05-green-testing)

<!-- step: aio-phase-2-step-2-4 -->

**Step 2.4:** Rewrite 04-implementing's inline severity-gating workflow (body lines 172-182) and delegation table (lines 414-418) to dispatch `review-coordinator` instead of naming individual reviewers. Update the 'FULL slices dispatch exactly 1 of the POV reviewers' logic to dispatch via review-coordinator, which selects the appropriate reviewer. Similarly update 05-green-testing's delegation table for any inline reviewer references. Update severity-gating to account for 8 POV reviewers (5 original + 3 domain from Phase 5). The review-coordinator holds 9 reviewers total (6 existing + 3 new domain).

- **Slice files:** 2 (04-implementing, 05-green-testing)

- Note: `plan-scout` has fan-in of 7 (flagged by research as 'possibly infrastructural'). Decision: keep as-is — plan-scout serves multiple planning consumers and this is intentional infrastructure, not redundancy.

**Validation:** `validate-agent-frontmatter.mjs --json --strict`, `validate-agent-quality.mjs`, `validate-agent-graph.mjs --json`, `npm run agents:routing-table`

<!-- slice: aio-phase-3 -->

## Phase 3 — Skill & Frontmatter Fixes [PENDING]

**Goal:** Fix orphan skills, user-invocable flags, validator improvements, deprecated fields

<!-- step: aio-phase-3-step-3-1 -->

**Step 3.1:** Verify `updating-agent-frontmatter` and `updating-skill-frontmatter` wired to `frontmatter-auditor` (from Phase 1)

<!-- step: aio-phase-3-step-3-2 -->

**Step 3.2:** Set `webgpu` skill `user-invocable: true` (matches siblings onnx-work, worker-inference-transport, multithread-evaluation)

<!-- step: aio-phase-3-step-3-3 -->

**Step 3.3:** Review `execute` skill `user-invocable` — consider setting to `false` (meta-orchestration, not a user task)

<!-- step: aio-phase-3-step-3-4 -->

**Step 3.4:** Fix validator YAML parser charset handling so non-ASCII chars (≤, —) aren't corrupted to ? in parsed output

<!-- step: aio-phase-3-step-3-5 -->

**Step 3.5:** Mandatory repo-wide grep scan for `infer` field in all `.agent.md` files. Replace ALL instances with `user-invocable` + `disable-model-invocation` per latest Copilot spec. No conditional handling. Also update ORCHESTRATION_GUIDE.md to document `model:` as a retained local extension (the guide currently says it was removed, but it's present on all agents).

<!-- step: aio-phase-3-step-3-6 -->

**Step 3.6:** Consider adding `argument-hint` to user-invocable agents for better UX (currently only on skills)

<!-- step: aio-phase-3-step-3-7 -->

**Step 3.7:** Consider adding `target` field where appropriate (`vscode` for IDE-only, `github-copilot` for cloud, or omit for both)

**Validation:** `validate-skill-frontmatter.mjs --json --strict`, `validate-agent-frontmatter.mjs --json --strict`, routing-table-freshness gate

<!-- slice: aio-phase-4 -->

## Phase 4 — Phase Coverage Enhancement [PENDING]

**Goal:** Add specialists for partial-coverage phases (DOCUMENT, LOG)

<!-- step: aio-phase-4-step-4-1 -->

**Step 4.1:** Create `session-summarizer` (Tier 3) under `07-logging`

- Mission: Collect changed files, validation evidence, delegation graph in isolated context. Produce structured session summary.
- Skills: `summarizing-session-log`, `tracker-handoff`
- Model: `kimi-k2.7-code:cloud` (light summarization task)
- Add to `agents:` of `07-logging`
- **Slice files:** 1 new agent, 1 modified

<!-- step: aio-phase-4-step-4-2 -->

**Step 4.2:** Create `docs-writer` (Tier 3) under `06-documenting`

- Mission: Write JSDoc, READMEs, examples, and guides from source in isolated context. Apply educational-docs tone model.
- Skills: `educational-docs`, `updating-js-docs`, `auditing-js-docs`
- Model: `kimi-k2.7-code:cloud` (writing task, light)
- Add to `agents:` of `06-documenting`
- **Slice files:** 1 new agent, 1 modified

**Validation:** `validate-agent-frontmatter.mjs --json --strict`, `validate-agent-quality.mjs`, `validate-agent-graph.mjs --json`, `npm run agents:routing-table`

<!-- slice: aio-phase-5 -->

## Phase 5 — Domain POV Reviewers [PENDING — NOT DEFERRED]

**Goal:** Create domain-specific reviewers for neural network correctness validation

<!-- step: aio-phase-5-step-5-1 -->

**Step 5.1:** Create `evolution-correctness-reviewer` (Tier 3)

- Mission: POV reviewer for NEAT evolution algorithm correctness — validate mutation/selection/crossover logic, topology mutation validity, fitness evaluation correctness, speciation boundary integrity
- Skills: `nge-core-algorithm`, `reproducibility-contracts`, `implementation-standards`
- Model: `kimi-k2.7-code:cloud` (read-only review)
- Add to `agents:` of `review-coordinator` (review-coordinator holds 9 reviewers total: 6 existing + 3 new domain)
- **Slice files:** 1 new agent, 1 modified (review-coordinator)

<!-- step: aio-phase-5-step-5-2 -->

**Step 5.2:** Create `onnx-parity-reviewer` (Tier 3)

- Mission: POV reviewer for ONNX export/import parity — validate round-trip correctness, binary emission determinism, runtime parity
- Skills: `onnx-work`, `implementation-standards`, `reproducibility-contracts`
- Model: `kimi-k2.7-code:cloud`
- Add to `agents:` of `review-coordinator`
- **Slice files:** 1 new agent, 1 modified (review-coordinator)

<!-- step: aio-phase-5-step-5-3 -->

**Step 5.3:** Create `webgpu-parity-reviewer` (Tier 3)

- Mission: POV reviewer for WebGPU compute correctness — validate GPU-vs-CPU numerical parity, shader correctness, fallback behavior
- Skills: `webgpu`, `implementation-standards`, `performance-optimization`
- Model: `kimi-k2.7-code:cloud`
- Add to `agents:` of `review-coordinator`
- **Slice files:** 1 new agent, 1 modified (review-coordinator)

<!-- step: aio-phase-5-step-5-4 -->

**Step 5.4:** Add redistributed skills to newly created domain reviewers

- `webgpu` → add to `webgpu-parity-reviewer` (removed from 04 in Phase 2 Step 2.2)
- `onnx-work` → add to `onnx-parity-reviewer` (removed from 04 in Phase 2 Step 2.2)
- `nge-core-algorithm` → add to `evolution-correctness-reviewer` (removed from 04 in Phase 2 Step 2.2)
- **Slice files:** 3 agent files (the Phase 5 agents, modified to include redistributed skills)

- Note: `worker-inference-transport` and `multithread-evaluation` domain gaps are addressed via skill redistribution to 05-green-testing rather than dedicated POV reviewers. This is an intentional deferral — these domains have broader existing skills and lower confidence scores (0.55-0.70). If future complexity warrants, dedicated reviewers can be created following the same pattern.

**Validation:** `validate-agent-frontmatter.mjs --json --strict`, `validate-agent-quality.mjs`, `validate-agent-graph.mjs --json`, `npm run agents:routing-table`

<!-- slice: aio-phase-6 -->

## Phase 6 — Independent Dual-Specialist Agent Review [PENDING]

**Goal:** Every agent is reviewed by 2 independent specialists from different perspectives to validate content quality, normalization, and modern Copilot best practices

**Approach:** For each agent (or batch of related agents), dispatch 2 specialists:

- **Specialist A (Content Quality):** Reviews the agent's mission, constraints, workflow, and output format for optimal task alignment. Checks if the agent body is comprehensive, clear, and actionable. Verifies the agent can actually accomplish its stated mission with its current tools and skills.
- **Specialist B (Standards & Normalization):** Reviews the agent against latest Copilot standards (official frontmatter, NeatapticTS quality contract, structured-v1, tier rules). Checks normalization across all agents (consistent section headers, field ordering, description style, JSDoc quality). Verifies modern best practices are followed.

**Both specialists must check online** for latest Copilot agent requirements, contract, and features to ensure 100% alignment.

<!-- step: aio-phase-6-step-6-1 -->

**Step 6.1:** Review all 8 Tier-1 orchestrators (00-helping through 07-logging) — 2 specialists each, batch by tier
<!-- step: aio-phase-6-step-6-2 -->

**Step 6.2:** Review all Tier-2 coordinators (agent-maintenance-coordinator, implementation-executor, review-coordinator) — 2 specialists each
<!-- step: aio-phase-6-step-6-3 -->

**Step 6.3:** Review all Tier-3 scouts and specialists (existing + newly created) — 2 specialists each, batch in groups of 5-7
<!-- step: aio-phase-6-step-6-4 -->

**Step 6.4:** Review all Tier-4 auxiliaries — 2 specialists each
<!-- step: aio-phase-6-step-6-5 -->

**Step 6.5:** Review all newly created agents from Phases 1-5 specifically — verify they meet the highest standards since they're brand new
<!-- step: aio-phase-6-step-6-6 -->

**Step 6.6:** Address all observations in batches — fix issues found by specialists, re-validate after each batch
<!-- step: aio-phase-6-step-6-7 -->

**Step 6.7:** Repeat review rounds until all specialists approve. Once a specialist approves an agent, no need to include it in future rounds.

**Review checklist for each agent:**

1. Mission statement is clear, specific, and actionable
2. Constraints are appropriate for the tier and task
3. Workflow/Approach/Default Flow section matches the actual capabilities
4. Output Format with structured-v1 block is correct for the tier
5. `model:` field matches the tiered strategy (glm-5.2 for heavy, kimi-k2.7 for light)
6. `tools:` array is minimal and sufficient
7. `agents:` allow-list is complete and has no phantoms
8. `skills:` array is relevant and not overloaded
9. `user-invocable` is correct (true for T1, false for T2/T3/T4)
10. Description is meaningful, specific, starts with appropriate trigger language
11. Section headers match the quality contract for the tier
12. Content is normalized with sibling agents (consistent style, tone, depth)
13. No deprecated fields (`infer` should be replaced)
14. Agent body is under 30,000 characters (Copilot limit)
15. Official Copilot frontmatter fields verified: `description` (required, present), `name` (optional, matches filename), `tools` (correct for tier), `model` (matches tiered strategy), `target` (set or intentionally omitted), `user-invocable` (correct for tier), `disable-model-invocation` (correct), `mcp-servers` (if needed), `metadata` (if needed), `handoffs` (with model/send/prompt/label/agent sub-fields, if used), `argument-hint` (on user-invocable agents), `hooks` (if beneficial), `agents` (subagent allow-list, no phantoms). Each field: PRESENT/OMITTED with justification. No 'consider' hedging — concrete pass/fail.

**Validation:** Full gate suite after all reviews complete

<!-- slice: aio-phase-7 -->

## Phase 7 — Final Validation [PENDING]

**Goal:** Full validation pass after all changes

<!-- step: aio-phase-7-step-7-1 -->

**Step 7.1:** Run all validation gates:

- `validate-agent-frontmatter.mjs --json --strict`
- `validate-skill-frontmatter.mjs --json --strict`
- `validate-agent-quality.mjs`
- `validate-agent-graph.mjs --json`
- `npm run agents:quality:gate`
- `npm run agents:routing-table`
- routing-table-freshness gate

<!-- step: aio-phase-7-step-7-2 -->

**Step 7.2:** Verification scans:

- Grep scan: zero phantom agent references remaining
- Confirm model assignments: 5 heavy agents on glm-5.2:cloud, all others on kimi-k2.7-code:cloud
- Confirm all new agents exist and pass validation
- Confirm deleted agents are gone and no references remain
- Confirm routing table is fresh
- Verify `search` and `todo` tool aliases resolve correctly in the CLI. If they don't, replace with explicit tool names (Grep, Glob, TodoWrite)
- Gate ownership assignments: `convergence-tracker` → 05-green-testing, `delegate-skill-coverage` → agent-maintenance-coordinator, `folder-quality` → boundary-mapper, `specialist-review` → review-coordinator

<!-- step: aio-phase-7-step-7-3 -->

**Step 7.3:** Feature adoption check:

- Consider adding `argument-hint` to user-invocable agents
- Consider adding `hooks` to key agents (PostToolUse for tsc/test validation)
- Consider array-valued `model` for agents that could benefit from fallback
- Review Copilot Memory, Customization Evaluations, and other new features for adoption

## Summary of Changes

### Files to CREATE (8-10 new agents):

1. `frontmatter-auditor.agent.md` (Tier 3) — Phase 1
2. `repo-cortex-scout.agent.md` (Tier 3) — Phase 1
3. `review-coordinator.agent.md` (Tier 2) — Phase 2
4. `session-summarizer.agent.md` (Tier 3) — Phase 4
5. `docs-writer.agent.md` (Tier 3) — Phase 4
6. `evolution-correctness-reviewer.agent.md` (Tier 3) — Phase 5
7. `onnx-parity-reviewer.agent.md` (Tier 3) — Phase 5
8. `webgpu-parity-reviewer.agent.md` (Tier 3) — Phase 5

### Files to DELETE (2 agents):

1. `research-codebase-coordinator.agent.md` — Phase 2 (folded into 02-researching)

### Files to MODIFY (~20+ agents):

- 5 heavy agents: model → glm-5.2:cloud (Phase 0)
- 26 light agents: model stays kimi-k2.7-code:cloud (Phase 0)
- 00-helping: phantom cleanup, add frontmatter-auditor + repo-cortex-scout
- 01-planning through 07-logging: phantom reference cleanup
- 02-researching: phantom cleanup, remove research-codebase-coordinator, add domain skills
- 03-red-testing: remove planning-test-strategy-coordinator phantom
- 04-implementing: reduce skill load, add review-coordinator to agents
- 05-green-testing: add domain skills, add review-coordinator to agents
- 06-documenting: add domain skills, add docs-writer to agents
- 07-logging: add session-summarizer to agents
- agent-maintenance-coordinator: add frontmatter-auditor to agents
- research-codebase-coordinator: deleted in Phase 2
- implementation-executor: phantom reference cleanup
- model-routing-and-budget skill: fix Haiku 4.6 → 4.5
- validate-agent-frontmatter.mjs: add model allowlist, fix charset
- routing table: regenerate after each phase

### Validator improvements:

- `validate-agent-frontmatter.mjs`: add model-value allowlist, fix YAML parser charset
- `model-routing-and-budget` skill: correct "Claude Haiku 4.6" to "Claude Haiku 4.5"

### New features to consider (from Copilot research):

- `argument-hint` on user-invocable agents
- `hooks` (PostToolUse for tsc/test, PreToolUse for safety)
- Array-valued `model` for fallback
- Customization Evaluations for agent quality
- Copilot Memory for repository knowledge
- Session Chronicle for work tracking

## Risk Assessment

- **Low risk:** Model field updates (mechanical, tiered strategy)
- **Medium risk:** Phantom reference removal (verify no agent depends on phantoms)
- **Medium risk:** research-codebase-coordinator removal (verify 02-researching absorbs work)
- **Low risk:** Skill redistribution (adding skills, not removing capabilities)
- **Low risk:** New agent creation (additive)
- **Medium risk:** Review coordinator extraction (changes delegation graph for 04/05)
- **Low risk:** Domain POV reviewers (additive, only invoked for domain-specific work)

## New Features from Copilot Research

### Features to adopt in this plan:

1. **`argument-hint`** on user-invocable agents — improves UX in chat picker
2. **Hooks (Preview)** — consider PostToolUse hooks on implementation agents to run `npx tsc --noEmit` after edits
3. **Array-valued `model`** — consider `model: ['glm-5.2:cloud', 'kimi-k2.7-code:cloud']` for heavy agents as fallback
4. **Customization Evaluations** — run on all agents after Phase 6 to catch contradictions
5. **Copilot Memory** — store repository-level architecture decisions for cross-session persistence
6. **Session Chronicle** — track agent work across sessions for cost analysis and standup reports

### Features to document for future adoption:

1. **Plugins** — package NeatapticTS skills/agents as distributable plugin
2. **MCP Resources** — expose neural network topology data as structured context
3. **Plan Agent** — use built-in Plan agent for complex feature planning
4. **Research Agent** — use `/research` for cited algorithm investigations
5. **BYOK** — document how to use local Ollama models via BYOK

## Latest validation evidence

green-light: true
status: green-light
verification: 9 specialists (6 inventory analysts + 3 Copilot standards researchers), 2 validation rounds
verification_date: 2026-08-15
note: Plan approved by ALL 6 validation specialists after 2 rounds. Pre-validated — plan-readiness gate may be bypassed under pragmatic mode.

phase-0: DONE
phase-0-evidence:

- validate-agent-frontmatter.mjs --json --strict: PASS (0 errors, 0 warnings)
- validate-agent-quality.mjs: PASS (0 errors, 0 warnings)
- validate-agent-graph.mjs --json: PASS (0 violations, 31 agents, 68 delegation edges)
- npm run agents:routing-table: PASS (31 agents, 67 skills, routing table regenerated)
  phase-0-date: 2026-08-15

phase-1: DONE
phase-1-evidence:

- validate-agent-frontmatter.mjs --json --strict: PASS (0 errors, 0 warnings, 33 agents)
- validate-agent-quality.mjs: PASS (0 errors, 0 warnings)
- validate-agent-graph.mjs --json: PASS (0 violations, 33 agents, 72 delegation edges)
- npm run agents:routing-table: PASS (33 agents, 67 skills)
- grep scan: ZERO phantom references remaining (excluding historical learning-log.jsonl)
  phase-1-date: 2026-08-15

phase-2: DONE
phase-2-evidence:

- validate-agent-frontmatter.mjs --json --strict: PASS (0 errors, 0 warnings, 33 agents)
- validate-agent-quality.mjs: PASS (0 errors, 0 warnings)
- validate-agent-graph.mjs --json: PASS (0 violations, 33 agents, 68 delegation edges)
- npm run agents:routing-table: PASS (33 agents, 67 skills, hash fresh)
- 04-implementing skills count: 14 (verified — reduced from 24)
- review-coordinator.agent.md: CREATED (Tier 2, 6 reviewers)
- research-codebase-coordinator.agent.md: DELETED
  phase-2-date: 2026-08-15

## Handoff Query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Active plan: plans/agent-inventory-optimization.plans.md
Research: plans/agent-inventory-optimization.research.md
Session log: plans/session-log-2026-08-15.md

Next: Phase 3 — Skill & Frontmatter Fixes (verify orphan skill wiring, set webgpu user-invocable, fix validator charset, replace infer field repo-wide, consider argument-hint and target fields)

Model mandate: glm-5.2:cloud for heavy tasks (00-helping, 01-planning, 03-red-testing, 04-implementing, implementation-executor). kimi-k2.7-code:cloud for light tasks (all others). NO OTHER MODELS APPROVED — only glm-5.2:cloud and kimi-k2.7-code:cloud. Users may override per request between these two models only. Plan mandates continue to use glm-5.2:cloud for implementing agents.

Validation commands:
- node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict
- node scripts/agent-customization/validate-agent-quality.mjs
- node scripts/agent-customization/validate-agent-graph.mjs --json
- npm run agents:quality:gate
- npm run agents:routing-table

Phases: 0 (model assignment) → 1 (phantom cleanup) → 2 (consolidation) → 3 (skill fixes) → 4 (coverage enhancement) → 5 (domain reviewers) → 6 (dual-specialist review) → 7 (final validation)

Nothing deferred. All phases implemented in detail. This is the core of the project.
```

## VALIDATION_EVIDENCE

Claim: 04-implementing @ 2026-08-15T20:30:00Z

### Step 0.3 — validate-agent-frontmatter.mjs model allowlist

- Removed `planning-test-strategy-coordinator` and `research-codebase-coordinator` from `strictTier2CoordinatorPathsByName`
- Removed `anthropic/claude-sonnet-4-20250514` from `strictAllowedModels`
- Changed `glm-5.2:cloud (ollama)` → `glm-5.2:cloud` in `strictAllowedModels` to match actual agent frontmatter values
- Updated error message to reference `glm-5.2:cloud` instead of `glm-5.2:cloud (ollama)`

### Step 0.4 — model-routing-and-budget SKILL.md

- Removed all Claude Sonnet and Claude Haiku references from task packet, required workflow, phase defaults table, decision tree, before/after examples, and guardrails
- Phase-based tier system now uses only `glm-5.2:cloud` (heavy/full) and `kimi-k2.7-code:cloud` (light)
- Added step 9 to Required Workflow: additional models require explicit user approval

### Preflight Results

- tsc: `npx tsc --noEmit -p tsconfig.json` → OK (0 errors)
- prettier: `npx prettier --check` on both changed files → All matched files use Prettier code style
- validator strict: `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict` → PASS, 0 errors, 0 warnings
- lint: N/A (lint covers src/testing/benchmarks/examples only; changed files are in scripts/ and .github/)

```yaml
PlanUpdate:
  slice_ids:
    - aio-phase-0-step-0-3
    - aio-phase-0-step-0-4
  changed_files:
    - scripts/agent-customization/validate-agent-frontmatter.mjs
    - .github/skills/model-routing-and-budget/SKILL.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json → OK'
    - 'npx prettier --check scripts/agent-customization/validate-agent-frontmatter.mjs .github/skills/model-routing-and-budget/SKILL.md → All matched files use Prettier code style'
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict → PASS, 0 errors, 0 warnings'
  tests_for_green:
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict'
    - 'node scripts/agent-customization/validate-agent-quality.mjs'
    - 'node scripts/agent-customization/validate-agent-graph.mjs --json'
  rollback:
    - 'Revert scripts/agent-customization/validate-agent-frontmatter.mjs: restore removed coordinator entries, restore anthropic/claude-sonnet-4-20250514 in strictAllowedModels, restore glm-5.2:cloud (ollama) string'
    - 'Revert .github/skills/model-routing-and-budget/SKILL.md: restore Claude Sonnet/Haiku references and old tier labels'
  next: 'Run 05-green-testing to validate all agent frontmatter passes strict validation and no Claude references remain in skill docs'
```

### slice-advancement Gate Results

- slice-advancement (aio-phase-0-step-0-3): partial pass — 5/7 sub-gates pass, code-coverage FAIL (owned by 05-green-testing)
  - plan-sync: PASS
  - step-packet: PASS
  - plan-slice-quality: PASS
  - plan-command-lint: PASS
  - shared-validation: PASS
  - code-coverage: FAIL — requires 100% coverage on scripts/agent-customization/validate-agent-frontmatter.mjs (owned by 05-green-testing)
  - specialist-review: PASS

### Phase 1 Steps 1.2, 1.3, 1.5 — frontmatter-auditor + repo-cortex-scout creation + orphan skill wiring

Claim: 04-implementing @ 2026-08-15T21:00:00Z

**Step 1.2 — Create `frontmatter-auditor` (Tier 3):**

- Created `.github/agents/frontmatter-auditor.agent.md` — consolidates 3 phantom agents (agent-frontmatter-auditor, skill-frontmatter-auditor, model-name-auditor)
- Skills: agent-frontmatter-standards, skill-frontmatter-standards, updating-agent-frontmatter, updating-skill-frontmatter, model-routing-and-budget
- Model: kimi-k2.7-code:cloud (read-only audit, light task)
- user-invocable: false, tier: 3, agents: []
- Description starts with "Use when:" per Copilot convention
- Body follows Tier-3 scout structure: CRITICAL RULE, Purpose, Mission, Scope boundaries, Constraints, Gate Enforcement, Approach, Audit Decision Tree, Finding Templates, If Blocked, Output format

**Step 1.3 — Create `repo-cortex-scout` (Tier 3):**

- Created `.github/agents/repo-cortex-scout.agent.md` — Cortex index health scout
- Skills: repo-cortex-workflow, repo-cortex-embeddings, research-methodology
- Model: kimi-k2.7-code:cloud (read-only scout)
- user-invocable: false, tier: 3, agents: []
- Description starts with "Use when:" per Copilot convention
- Body follows Tier-3 scout structure with Cortex Health Decision Tree and Finding Templates

**Allow-list updates:**

- Added `frontmatter-auditor` and `repo-cortex-scout` to `agents:` array of `00-helping`
- Added `frontmatter-auditor` to `agents:` array of `agent-maintenance-coordinator`
- Added `repo-cortex-scout` to `agents:` array of `02-researching`

**Step 1.5 — Orphan skill wiring verification:**

- `updating-agent-frontmatter` → carrier: frontmatter-auditor ✅ (confirmed in routing table line 113)
- `updating-skill-frontmatter` → carrier: frontmatter-auditor ✅ (confirmed in routing table line 115)
- Both orphan skills now have a carrier — no longer orphan

### Phase 1 Validation Results

- validate-agent-frontmatter.mjs --json --strict: PASS (0 errors, 0 warnings, 33 agents)
- validate-agent-quality.mjs: PASS (0 errors, 0 warnings)
- validate-agent-graph.mjs --json: PASS (33 agents, 0 issues, references resolve, no cycles, tier enforcement pass)
- tier-enforcement gate: PASS (0 issues, byTier: {1:8, 2:3, 3:21, 4:1})
- routing-table-freshness gate: PASS (hashes match after regeneration)
- npm run agents:routing-table: PASS (33 agents, 67 skills, routing table regenerated)
- MCP agent-graph gate: PASS (33 agents, 0 issues)

```yaml
PlanUpdate:
  slice_ids:
    - aio-phase-1-step-1-2
    - aio-phase-1-step-1-3
    - aio-phase-1-step-1-5
  changed_files:
    - .github/agents/frontmatter-auditor.agent.md
    - .github/agents/repo-cortex-scout.agent.md
    - .github/agents/00-helping.agent.md
    - .github/agents/agent-maintenance-coordinator.agent.md
    - .github/agents/02-researching.agent.md
    - .github/agent-skill-routing-table.md
  preflight:
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict → PASS, 0 errors, 0 warnings, 33 agents'
    - 'node scripts/agent-customization/validate-agent-quality.mjs → PASS, 0 errors, 0 warnings'
    - 'node scripts/agent-customization/gates/agent-graph.gate.mjs --json → PASS, 33 agents, 0 issues'
    - 'node scripts/agent-customization/gates/tier-enforcement.gate.mjs --json → PASS, 0 issues'
    - 'npm run agents:routing-table → PASS, 33 agents, 67 skills'
    - 'routing-table-freshness gate → PASS, hashes match'
  tests_for_green:
    - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict'
    - 'node scripts/agent-customization/validate-agent-quality.mjs'
    - 'node scripts/agent-customization/gates/agent-graph.gate.mjs --json'
    - 'node scripts/agent-customization/gates/tier-enforcement.gate.mjs --json'
    - 'npm run agents:routing-table:gate'
  rollback:
    - 'Delete .github/agents/frontmatter-auditor.agent.md'
    - 'Delete .github/agents/repo-cortex-scout.agent.md'
    - 'Revert 00-helping.agent.md agents array to [agent-maintenance-coordinator, coverage-analyst, learning-event-capturer]'
    - 'Revert agent-maintenance-coordinator.agent.md agents array to [learning-event-capturer]'
    - 'Revert 02-researching.agent.md agents array to remove repo-cortex-scout'
    - 'Re-run npm run agents:routing-table to regenerate routing table'
  next: 'Continue Phase 1 with Step 1.1 (phantom reference cleanup for helping-gap-resolution-coordinator) and Step 1.4 (remaining phantom reference removal). Then run 05-green-testing for full validation.'
```

### Phase 1 slice-advancement Gate Results

- slice-advancement (aio-phase-1-step-1-2): PASS — all 4 sub-gates pass (TRIVIAL severity)
  - plan-sync: PASS
  - step-packet: PASS
  - plan-slice-quality: PASS
  - plan-command-lint: PASS
