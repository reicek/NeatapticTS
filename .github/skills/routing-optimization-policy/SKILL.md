---
name: routing-optimization-policy
description: 'Use when: enforcing routing discipline, tier boundaries, or routing-table freshness.'
argument-hint: 'Describe the routing decision, tier boundary, or gate protocol to validate. Include target agent/skill, suspected violation, and validation command.'
user-invocable: false
disable-model-invocation: false
tools: [neataptic-gate-mcp/*, neataptic-workflow-mcp/*]
skills:
  - execute
  - agent-frontmatter-standards
  - green-validation-gates
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Routing Optimization Policy

Use this skill when routing discipline needs enforcement, validation, or audit across the eight numbered SDLC orchestrators and their delegated specialists.

This skill owns the durable knowledge surface for routing policy in this repository. It defines tier boundaries, delegation contracts, flow-and-gate protocols, and skill-companion routing rules. Companion agents such as `Plan Scout` or `Implementation-Pattern Scout` should gather evidence and hand specific routing questions into this skill rather than restating routing philosophy.

## When to Use

- A main agent attempt is performing substantive work (implementation, research, testing, documentation, logging) without delegating to the smallest relevant numbered orchestrator.
- A Tier 2 coordinator or Tier 3 scout is delegating to a higher-numbered tier without using `00.cross-tier-helper`.
- A numbered SDLC agent is executing body work without selecting a named flow from `.github/flows/`.
- A flow is completing without running its declared exit gates or without recording gate exceptions via `record_gate_exception`.
- The canonical routing table at `.github/agent-skill-routing-table.md` is stale or has not been validated via `npm run agents:routing-table:gate`.
- A `.github/agents/*.agent.md` file is missing the `skills: [...]` frontmatter field.
- A skill and companion agent overlap in ownership; the agent needs updating to follow the skill's durable policy.
- Gate failures are being routed back for continued repair without an artificial retry threshold.
- A dispatch overlaps another live dispatch on the same file, or a single dispatch owns too many files (Doctrine Rule 1).
- Independent scouts are being dispatched one after another instead of in parallel (Doctrine Rule 2).
- Findings are being merged in more than one place, or agents are chaining results to peers (Doctrine Rule 3).
- Two write-capable agents are dispatched over overlapping file sets (Doctrine Rule 4).
- An agent exceeded its time budget without stopping to return a gap report (Doctrine Rule 5).

## When NOT to use

Do NOT use for simple routing decisions that `execute` can handle directly. Do NOT use for frontmatter validation - use `agent-frontmatter-standards` instead.

## Workflow Diagram

```text
Flowchart summary: "Routing request" → "Classify surface"; "Classify surface" → "Tier boundary?"; "Tier boundary?" → "Check delegation direction" (Yes), "Check gate or freshness" (No); "Check delegation direction" → "Dispatch to correct tier"; "Check gate or freshness" → "Run validation gate"; "Dispatch to correct tier" → "Record evidence"; "Run validation gate" → "Record evidence"; "Record evidence" → "Done"; "Done".
```

## Tier Graph Contract

The repository enforces a strict five-tier agent graph. Delegation must flow downward except via the cross-tier helper.

| Tier | Label                                  | User-Invocable | Examples                                                                                                                               |
| ---- | -------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | Default / Main                         | false          | Default VS Code Copilot agent                                                                                                          |
| 1    | Numbered SDLC Orchestrators            | true           | `00-helping`, `01-planning`, `02-researching`, `03-red-testing`, `04-implementing`, `05-green-testing`, `06-documenting`, `07-logging` |
| 2    | Named Coordinators / Sub-Orchestrators | false          | `planning-context-coordinator`, `solid-split`                                                                                          |
| 3    | Hidden Scouts and Specialists          | false          | `Boundary Mapper`, `Coverage Scout`, `Plan Scout`, `Docs Scout`                                                                        |
| 4    | Auxiliaries and One-Shot Helpers       | false          | `acceptance-criteria-writer`, `file-change-summarizer`                                                                                 |

### Delegation Rules

- **Tier 1** may delegate to Tier 2, 3, or 4.
- **Tier 2** may delegate to Tier 3 or 4.
- **Tier 3** may delegate to Tier 4 only.
- **Tier 4** may not delegate to any agent.
- **No tier** may call a higher-numbered tier except via `00.cross-tier-helper`.
- **User-invocable: true** is valid only for Tier 1 agents.

### Mission Rule

The main agent (Tier 0) must not perform implementation, refactoring, research, planning, testing, documentation, or logging directly. All substantive work routes to the smallest relevant numbered SDLC orchestrator (Tier 1).

**Exceptions:**

- Trivial factual answers (single-sentence lookups, no file changes).
- Direct operator commands with zero file reads/writes.

## Orchestration Doctrine

The `execute` skill (§9.0) is the canonical statement of orchestration
doctrine. This skill **audits compliance** with it; it does not restate a
second, divergent copy of the mechanics. The five rules are binding for
every dispatch in the repository:

| #   | Rule                   | Requirement                                                                                                                   | Audited by                        |
| --- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| 1   | **Thin scope**         | Every dispatched agent gets exactly **one** objective and the **minimum** file set. No two live dispatches own the same file. | §9.1 "Thin Scope Before Dispatch" |
| 2   | **Parallel scouts**    | Independent scouts (read-only, mutually independent questions) MUST be dispatched **in parallel**.                            | §9.2 "Parallel Scouts"            |
| 3   | **Central merge**      | Findings merge at **one** central point — the orchestrator. No agent-to-agent chaining; scouts report, not decide.            | §9.2 "Central Merge"              |
| 4   | **Parallel executors** | Write-capable agents may run in parallel **only** over **non-overlapping file sets**; overlap is serialized.                  | §9.2 "Parallel Executors"         |
| 5   | **Time-budget exit**   | An agent exceeding its time budget **stops** and returns a **gap report** for a fresh takeover.                               | §9.1 "Gap Report Contract"        |

### Doctrine-to-Tier Mapping

| Rule                   | Applies to                                | Violation symptom                                                     |
| ---------------------- | ----------------------------------------- | --------------------------------------------------------------------- |
| 1 — Thin scope         | Every dispatch, Tier 1 → 4                | Two dispatches touch the same file; a dispatch owns a whole directory |
| 2 — Parallel scouts    | Tier 1 → Tier 3 read-only scouts          | Independent scouts dispatched one after another                       |
| 3 — Central merge      | Every orchestrator and Tier-2 coordinator | Findings merged in more than one place; sideways chaining             |
| 4 — Parallel executors | Tier 2/3/4 write-capable agents           | Two agents writing the same file concurrently                         |
| 5 — Time-budget exit   | Every dispatched agent                    | Agent overruns silently; unfinished work is unowned                   |

### Violation Handling

A doctrine violation is a **routing defect**, not a stylistic nit:

1. Record the violation with the specific dispatch that broke the rule.
2. Correct it in place — re-split an overlapping scope, re-dispatch
   independent scouts in parallel, or stop and re-dispatch an overrunning
   agent with a fresh instance seeded by its gap report.
3. Do NOT apply an artificial retry threshold: continue correcting until
   resolved or a true technical limit is reached.

## Flow and Gate Protocol

Every numbered SDLC agent selects a named flow from `.github/flows/` to execute body work. Each flow declares exit gates that must return a structured JSON contract before completion.

### Gate Contract

```json
{
  "pass": true,
  "evidence": "<command output or file path>",
  "fixHint": "<short remediation guidance>",
  "owner": "<agent or skill owning the fix>"
}
```

### Gate Exception Handling

- Gate failures are recorded via `scripts/agent-customization/gates/record-gate-exception.mjs`.
- Exceptions append to `.github/ai-learning/learning-log.jsonl`.
- Gate failures are recorded for audit and routed back for continued repair without an artificial retry threshold.

### Post-Phase Fanout

After flow body completes, run post-phase fanout gates to confirm downstream readiness before advancing to the next phase or step.

## Skill and Companion Routing

Skills and companion agents have distinct ownership boundaries.

### Skills Own Durable Knowledge

- Workflow standards and guardrails.
- Tone models and source-mapping rules.
- Validation expectations and handoff contracts.
- Citation, attribution, and Mermaid diagram policies.

### Companion Agents Are Thin and Task-Shaped

- Gather evidence and map boundaries.
- Scout drift and execute one workflow step.
- Defer durable policy to skills.

### Overlap Rule

When a skill and companion agent overlap in ownership, update the agent to follow the skill. Do not allow companion agents to restate durable policy that lives in a skill.

## Canonical Routing Table

The routing table is the authoritative catalog of all agents and skills.

- **Location:** `.github/agent-skill-routing-table.md`
- **Refresh Command:** `npm run agents:routing-table`
- **Validate Freshness:** `npm run agents:routing-table:gate` or `node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json`
- **Skills Field:** Every `.github/agents/*.agent.md` file must declare a `skills: [...]` frontmatter field.

## Required Workflow

1. **Identify the routing surface.**
   - Determine which agent, skill, or tier boundary is in scope.
   - Confirm whether the issue is delegation, gate protocol, or routing-table freshness.

2. **Validate tier enforcement.**
   - Confirm no Tier 2/3/4 agent is delegating upward without `00.cross-tier-helper`.
   - Confirm Tier 0 is not performing substantive work without routing to Tier 1.

3. **Audit orchestration doctrine.**
   - **Thin scope:** confirm each live dispatch owns one objective and a
     minimal file set, with zero file overlap between live dispatches.
   - **Parallel scouts:** confirm mutually independent read-only scouts were
     dispatched in parallel, not sequentially.
   - **Central merge:** confirm findings merge at exactly one central point
     (the orchestrator) with no agent-to-agent chaining.
   - **Parallel executors:** confirm parallel write dispatches cover
     non-overlapping file sets, and that overlapping sets were serialized
     through the file-lock tracker.
   - **Time-budget exit:** confirm any agent that exceeded its budget
     stopped and returned a gap report, and that a fresh instance took over
     from that report.
   - Reference: `execute` skill §9.0 (Canonical Orchestration Doctrine),
     §9.1, §9.2.

4. **Check flow-and-gate compliance.**
   - Verify the numbered agent selected a named flow from `.github/flows/`.
   - Verify exit gates ran and returned the structured contract.
   - If gates failed, confirm exceptions were recorded and routed back for
     continued repair without an artificial retry threshold. Continue retrying
     until resolved or a true technical limit is reached. No concessions.

5. **Audit routing-table freshness.**
   - Run `npm run agents:routing-table:gate` to confirm the routing table reflects current agent/skill inventory.
   - If stale, run `npm run agents:routing-table` to regenerate.

6. **Verify skill frontmatter.**
   - Confirm every `.github/agents/*.agent.md` has a `skills: [...]` field.
   - If missing, update the agent frontmatter to declare its skill dependencies.

7. **Resolve skill-companion overlap.**
   - If a companion agent restates durable policy owned by a skill, update the agent to defer to the skill.
   - Record the alignment fix in the agent file.

8. **Report the routing audit.**
   - Document which tier boundaries were validated.
   - List any violations found and remediation taken.
   - Confirm orchestration-doctrine compliance, routing-table freshness, and
     gate protocol compliance.

## Examples

### Example 1: Main Agent Routing Violation

**Scenario:** The main agent begins editing `src/neat/evolve/neat.evolve.ts` without delegating to `04-implementing`.

**Correct Action:**

```text
Use routing-optimization-policy to enforce Tier 0 → Tier 1 routing.
Violation: main agent editing production code without delegation.
Expected: route to 04-implementing with compact implementation packet.
Validate: confirm 04-implementing selected a flow and ran exit gates.
```

### Example 2: Cross-Tier Delegation Without Helper

**Scenario:** `solid-split` (Tier 2) directly invokes `01-planning` (Tier 1) to author a new step packet.

**Correct Action:**

```text
Use routing-optimization-policy to enforce tier graph discipline.
Violation: Tier 2 calling Tier 1 without 00.cross-tier-helper.
Expected: solid-split records blocker, routes via 00.cross-tier-helper, receives resolution summary.
Validate: confirm learning event logged to .github/ai-learning/learning-log.jsonl.
```

### Example 3: Missing Skills Frontmatter

**Scenario:** `boundary-mapper.agent.md` lacks a `skills: [...]` field.

**Correct Action:**

```text
Use routing-optimization-policy to audit agent frontmatter.
File: .github/agents/boundary-mapper.agent.md
Issue: missing skills field.
Expected addition: skills: ['solid-split', 'plan-alignment']
Validate: node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent=boundary-mapper
```

### Example 4: Gate Failure Without Escalation

**Scenario:** A flow in `05-green-testing` has failed its exit gate twice, but no escalation occurred.

**Correct Action:**

```text
Use routing-optimization-policy to enforce gate protocol.
Agent: 05-green-testing
Gate failures: recorded, no escalation.
Expected: continue retrying the gate or routing the failure back for repair
until resolved or a true technical limit is reached. No artificial threshold.
Validate: grep .github/ai-learning/learning-log.jsonl for gate_exception entries.
```

### Example 5: Overlapping Dispatch Scopes

**Scenario:** `04-implementing` and `docs-writer` are dispatched at the same
time; both own `src/neat/index.ts`.

**Correct Action:**

```text
Use routing-optimization-policy to enforce Doctrine Rule 1 and Rule 4.
Violation: two live dispatches own the same file.
Expected: split the scope so each dispatch owns a disjoint file set, then
re-dispatch in parallel; if the overlap cannot be split, serialize via the
file-lock tracker.
Validate: confirm no two live dispatches share a file path.
```

### Example 6: Agent Overran Its Time Budget

**Scenario:** A `04-implementing` instance has run past cap+50% with a static
`tool_calls_completed` and has not returned.

**Correct Action:**

```text
Use routing-optimization-policy to enforce Doctrine Rule 5.
Violation: agent exceeded its time budget without a gap report.
Expected: stop the agent, capture its gap report (completed / remaining /
blockers / next_action), and dispatch a FRESH instance against the same slice
seeded with that report.
Validate: confirm the plan records the gap report and the takeover dispatch.
```

## Coordination

- Use `agent-inventory-auditor` to discover tier violations across the full agent graph.
- Use `helping-agent-maintenance-coordinator` when agent frontmatter needs repair.
- Use `00-helping` when a routing gap reveals a missing specialist or weak skill.

## Decision Tree

```text
Flowchart summary: "Routing question" → "What kind?"; "What kind?" → "Use execute skill" (Simple delegation), "Use routing-optimization-policy" (Tier/gate enforcement), "Use agent-frontmatter-standards" (Frontmatter validation), "Run npm run agents:routing-table:gate" (Routing table freshness); "Use execute skill"; "Use routing-optimization-policy"; "Use agent-frontmatter-standards"; "Run npm run agents:routing-table:gate".
```

## Guardrails

- Do not allow Tier 0 to perform substantive work without routing to Tier 1.
- Do not allow upward tier delegation without `00.cross-tier-helper`.
- Do not merge a change that bypasses flow-and-gate protocol.
- Do not allow a skill and companion agent to maintain overlapping durable policy; align the agent to the skill.
- Do not treat a stale routing table as acceptable; regenerate and validate before marking routing work complete.
- Do not skip gate exception logging; every gate failure must append to `.github/ai-learning/learning-log.jsonl`.
- Do not dispatch an agent with an overlapping or oversized scope; split it first (Doctrine Rule 1).
- Do not dispatch mutually independent scouts sequentially; parallelize them (Doctrine Rule 2).
- Do not merge findings in more than one place; the orchestrator is the single merge point (Doctrine Rule 3).
- Do not co-write a file from two live dispatches; serialize overlapping file sets (Doctrine Rule 4).
- Do not let an overrunning agent grind past its budget; it must stop and return a gap report (Doctrine Rule 5).

## Expected Final Output

A strong routing-optimization pass should produce:

- confirmation that Tier 0 routed all substantive work to the smallest relevant Tier 1 orchestrator,
- validation that no Tier 2/3/4 agent delegated upward without `00.cross-tier-helper`,
- validation that the five orchestration-doctrine rules hold — thin non-overlapping scopes, parallel independent scouts, single central merge point, parallel executors over disjoint file sets, and time-budget exit via gap report,
- evidence that numbered agents selected named flows and ran exit gates with structured JSON contracts,
- confirmation that gate exceptions were logged and routed back for continued
  repair without an artificial retry threshold,
- routing-table freshness validation via `npm run agents:routing-table:gate`,
- updated agent frontmatter with `skills: [...]` fields where missing,
- alignment fixes where companion agents overlapped with skill-owned durable policy.
