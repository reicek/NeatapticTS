---
name: subagent-delegation-patterns
description: 'Use when: preparing or reviewing subagent task packets and delegation.'
argument-hint: 'Describe the parent phase, specialist needed, task packet, expected output format, and whether calls can run in parallel.'
user-invocable: false
disable-model-invocation: false
skills:
  - execute
  - creating-specialist-agent
  - splitting-monolithic-agent
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Subagent Delegation Patterns

Use this skill when a phase agent needs specialist help and must decide how to
structure the delegation — what to isolate, what to parallelize, and what to
require back.

This skill owns the workflow for constructing safe, self-contained specialist
task packets, deciding sequential versus parallel dispatch, and validating
specialist output before trusting it.

## When to Use

- A phase agent's next step requires narrow domain knowledge that would
  benefit from context isolation in a dedicated specialist call.
- Two or more independent reconnaissance tasks can run in parallel without
  ordering constraints.
- A specialist must stay read-only and the parent agent needs to constrain its
  tool scope explicitly.
- The parent agent needs to validate specialist output before writing to a
  plan or tracker.
- A prior delegation returned incomplete or wrong evidence and the packet needs
  to be narrowed and retried.
- A hidden specialist agent is available and assigning it improves coherence or
  cost efficiency.

## When NOT to use

Do NOT use for simple 1-2 read tasks that can be done directly with grep/view. Do NOT use for tasks that dont need delegation at all.

## Workflow Diagram

```text
Flowchart summary: "Task to delegate" → "Independent?"; "Independent?" → "Launch in parallel" (Yes), "Sequential dispatch" (No); "Launch in parallel" → "Collect all results"; "Sequential dispatch" → "Wait for each result"; "Collect all results" → "Merge findings"; "Wait for each result" → "Merge findings"; "Merge findings" → "Done"; "Done".
```

## Task Packet

Pass a compact packet describing the specialist, the bounded task, and the
required return format.

```text
Use subagent-delegation-patterns for Phase 2 boundary recon.
Parent phase: 02-research.
Specialist: Boundary Mapper (read-only discovery).
Task: identify all public surfaces in src/multithreading that cross the worker boundary.
Constraints: read-only; no edits; bounded to src/multithreading and its tests.
Return: list of crossing surfaces, their current transport type, and any coverage gaps.
Parallel eligible: yes — can run alongside the plan-read step.
```

## Required Workflow

1. Delegate only when context isolation or a narrower tool set improves the
   result over continuing in the parent context.
2. Keep each specialist task packet self-contained: goal, files, constraints,
   output format, and no-edit or read-only limits.
3. Use parallel subagents only for independent read-only discovery where
   ordering does not matter.
4. Use sequential subagents when later work depends on earlier evidence or plan
   state produced by a prior specialist.
5. Require specialists to return compact evidence and next-step recommendations,
   not full workflows owned by skills.
6. Validate specialist output against the required return fields before
   trusting it as input to the next step.
7. Record durable decisions in the active tracker when one exists, or in the
   learning log for reusable customization improvements.

## Specialist Packet Template

```text
Role: <hidden specialist or agent name>
Task: <one narrow objective>
Files or plans: <bounded list>
Constraints: <read-only/edit/validation limits>
Return: <exact output fields>
```

## Parallel vs Sequential Decision Tree

```text
Flowchart summary: "Multiple sub-tasks" → "Independent?"; "Independent?" → "Parallel: launch all at once" (Yes), "Sequential: wait for each" (No); "Parallel: launch all at once" → "Collect results in order"; "Sequential: wait for each" → "Pass result to next agent"; "Collect results in order" → "Merge"; "Pass result to next agent" → "Merge"; "Merge".
```

## Before / After Examples

**Before:**

```text
Task: look into the multithreading stuff and tell me what you find.
```

**After:**

```text
Role: boundary-mapper
Task: identify all public surfaces in src/multithreading that cross the worker boundary
Files: src/multithreading/**, testing/multithreading/**
Constraints: read-only; no edits
Return: list of crossing surfaces with transport type and coverage gaps
```

## Guardrails

- Do not delegate a task that the parent phase agent can resolve with one or
  two reads; delegation adds overhead and should have a clear isolation benefit.
- Do not run parallel subagents when the second task depends on output from the
  first; make the dependency explicit and run sequentially.
- Do not allow a specialist to make tracker or plan edits unless the parent
  agent has explicitly confirmed the edit scope.
- Do not accept specialist output that is missing required return fields; narrow
  the packet and retry.
- Do not embed full skill workflows inside a delegation packet; reference the
  relevant skill by name and let the specialist invoke it.

## Expected Final Output

A strong delegation pass should produce:

- a completed specialist task packet in the template shape,
- the specialist's compact evidence response with all required return fields,
- the parent agent's decision about which evidence to carry forward,
- any tracker or plan update reflecting the new information.
