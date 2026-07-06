---
name: agent-script-tooling
description: 'Use when: designing or refactoring reusable agent-customization scripts.'
argument-hint: 'Describe the script purpose, inputs, outputs, failure modes, and whether it reads, validates, or changes files.'
user-invocable: false
disable-model-invocation: false
skills:
  - agent-frontmatter-standards
  - green-validation-gates
  - implementation-standards
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Agent Script Tooling

Use this skill before adding or changing scripts that support agents or skills,
particularly anything under `scripts/agent-customization/` or
`rag-index/`.

This skill owns the design standards for agent-supporting Node ES module
scripts: interface shape, output format, error conventions, idempotency rules,
and validation exit codes.

## When to Use

- A new validation gate, freshness check, or health-summary script is being
  added under `scripts/agent-customization/` or `rag-index/`.
- An existing agent script needs `--json` output, `--help` text, or structured
  stderr diagnostics added.
- A script is being promoted from ad hoc to a durable gate used in step packet
  validation fields.
- A script's exit codes need to be audited to ensure callers can branch by
  failure type reliably.
- A new script needs dry-run or audit-mode behavior before it can be used in
  strict gate context.
- An existing script produces output that agents cannot parse reliably and needs
  a structured output contract.

## When NOT to use

Do NOT use for scripts outside the `scripts/agent-customization/` directory - this skill only covers agent customization tooling.

## Workflow Diagram

```text
Flowchart summary: "Script change" → "Run script directly"; "Run script directly" → "Exit code?"; "Exit code?" → "Check stderr for warnings" (0), "Fix reported error" (Non-zero); "Check stderr for warnings" → "Run validate-agent-frontmatter"; "Fix reported error" → "Run script directly"; "Run validate-agent-frontmatter" → "Pass?"; "Pass?" → "Done" (Yes), "Fix validation issue" (No); "Done"; "Fix validation issue" → "Run validate-agent-frontmatter".
```

## Task Packet

Pass a compact packet describing the script's purpose, the interface required,
and how it will be called.

```text
Use agent-script-tooling for a new freshness-check gate.
Purpose: validate that the semantic index age is under threshold and return structured evidence.
Interface: --json flag for machine output; --help for self-documentation; nonzero exit when stale.
Called by: cortex-index.gate.mjs via step packet validation fields.
Failure modes: index missing, over-age, or MCP unreachable.
Read or write: read-only.
```

## Required Workflow

1. Prefer self-contained Node ES modules with no dependency churn unless a
   parser or SDK is truly needed; plain `node:fs`, `node:path`, and
   `node:child_process` cover most cases.
2. Provide `--help` for every script that will be called by an agent or gate.
3. Support `--json` for validation and reporting scripts so agents can parse
   output without text scraping.
4. Write parseable data to stdout and diagnostics to stderr so callers can
   separate structured results from human-readable context.
5. Use noninteractive inputs only: flags, environment variables, stdin, or
   files. Never prompt.
6. Make validation scripts idempotent and safe to retry without side effects.
7. Use distinct nonzero exit codes only when callers need to branch by failure
   type; otherwise a single nonzero exit for all failures is sufficient.
8. Run the script in audit or dry-run mode before wiring it into strict gates.
9. Document the script's purpose, flags, output shape, and exit codes in a
   top-of-file JSDoc comment so `--help` can surface useful text.

## Interface Standards

### Required for gate scripts

- `--json`: emit `{pass: boolean, evidence: string, fixHint: string, owner: string}` to stdout.
- `--help`: emit usage, flags, exit codes, and a one-line description.
- Exit 0 on pass, nonzero on failure. Use distinct codes if callers branch.

### Required for reporting scripts

- `--json`: emit a structured summary object to stdout.
- Diagnostics (progress, warnings) to stderr only.
- Idempotent: repeated runs return the same result for the same repo state.

### Optional but preferred

- `--dry-run`: show what would change without writing anything.
- `--verbose`: additional diagnostic detail to stderr.

## Exit Code Reference

| Exit Code | Meaning                 | Action               |
| --------- | ----------------------- | -------------------- |
| 0         | Success, no issues      | Proceed              |
| 1         | Validation errors found | Fix reported issues  |
| 2         | Usage or argument error | Check command syntax |

## Decision Tree

```text
Flowchart summary: "Need new script" → "Purpose?"; "Purpose?" → "Create gate script (--json, exit codes)" (Block on pass/fail), "Create reporting script (--json summary)" (Summarize state), "Create utility script (--help)" (One-off helper); "Create gate script (--json, exit codes)" → "Audit in dry-run, then wire to gate"; "Create reporting script (--json summary)" → "Audit in dry-run, then wire to gate"; "Create utility script (--help)" → "Document in top-of-file JSDoc"; "Audit in dry-run, then wire to gate"; "Document in top-of-file JSDoc".
```

## Before / After Examples

**Before:**

```js
// ad hoc: no --help, prose on stdout, no exit code contract
console.log('Index is ' + (stale ? 'stale' : 'fresh'));
```

**After:**

```js
/**
 * Validate semantic index freshness. Exit 0 if fresh, 1 if stale.
 * @example node validate-index.mjs --json
 */
if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ pass: !stale, evidence, fixHint, owner }));
}
process.exit(stale ? 1 : 0);
```

## Guardrails

- Do not write interactive prompts or blocking stdin reads into scripts called
  by agents; agents run in noninteractive contexts.
- Do not mix structured JSON output and human-readable prose on stdout; pick one
  per invocation mode.
- Do not rely on side-effect order or global state for idempotency; make the
  script safe to interrupt and retry at any step.
- Do not promote a script to a gate before auditing it in dry-run mode at least
  once against the real repo.
- Do not add external npm dependencies for convenience; keep agent scripts
  self-contained so they work immediately after clone.

## Expected Final Output

A strong agent-script-tooling pass should produce:

- a script that passes `--help` and `--json` smoke checks,
- a documented exit code table in the script's top-of-file comment,
- a dry-run or audit-mode run result confirming behavior before gate wiring,
- updated step packet validation fields referencing the new script path and
  expected exit code.
