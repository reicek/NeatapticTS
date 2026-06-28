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
- Three consecutive gate failures have occurred in a session without escalation to `00-helping`.

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
| 2    | Named Coordinators / Sub-Orchestrators | false          | `planning-context-coordinator`, `solid-split`, `helping-gap-resolution-coordinator`                                                    |
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
- Three consecutive gate failures in a session trigger automatic escalation to `00-helping` via `00.cross-tier-helper`.

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

3. **Check flow-and-gate compliance.**
   - Verify the numbered agent selected a named flow from `.github/flows/`.
   - Verify exit gates ran and returned the structured contract.
   - If gates failed, confirm exceptions were recorded and escalation occurred after three consecutive failures.

4. **Audit routing-table freshness.**
   - Run `npm run agents:routing-table:gate` to confirm the routing table reflects current agent/skill inventory.
   - If stale, run `npm run agents:routing-table` to regenerate.

5. **Verify skill frontmatter.**
   - Confirm every `.github/agents/*.agent.md` has a `skills: [...]` field.
   - If missing, update the agent frontmatter to declare its skill dependencies.

6. **Resolve skill-companion overlap.**
   - If a companion agent restates durable policy owned by a skill, update the agent to defer to the skill.
   - Record the alignment fix in the agent file.

7. **Report the routing audit.**
   - Document which tier boundaries were validated.
   - List any violations found and remediation taken.
   - Confirm routing-table freshness and gate protocol compliance.

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
Gate failures: 2 recorded, no escalation.
Expected: on third consecutive failure, escalate to 00-helping via 00.cross-tier-helper.
Validate: grep .github/ai-learning/learning-log.jsonl for gate_exception entries.
```

## Coordination

- Use `agent-inventory-auditor` to discover tier violations across the full agent graph.
- Use `helping-agent-maintenance-coordinator` when agent frontmatter needs repair.
- Use `helping-gap-resolution-coordinator` when a routing gap reveals a missing specialist or weak skill.

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

## Expected Final Output

A strong routing-optimization pass should produce:

- confirmation that Tier 0 routed all substantive work to the smallest relevant Tier 1 orchestrator,
- validation that no Tier 2/3/4 agent delegated upward without `00.cross-tier-helper`,
- evidence that numbered agents selected named flows and ran exit gates with structured JSON contracts,
- confirmation that gate exceptions were logged and escalation occurred after three consecutive failures,
- routing-table freshness validation via `npm run agents:routing-table:gate`,
- updated agent frontmatter with `skills: [...]` fields where missing,
- alignment fixes where companion agents overlapped with skill-owned durable policy.
