---
name: agent-script-tooling
description: 'Design reusable scripts for NeatapticTS agent customization workflows. Use when adding scripts under scripts/agent-customization, defining --help, JSON output, stderr diagnostics, dry-run behavior, idempotency, or validation exit codes.'
argument-hint: 'Describe the script purpose, inputs, outputs, failure modes, and whether it reads, validates, or changes files.'
user-invocable: false
disable-model-invocation: false
---

# Agent Script Tooling

Use this skill before adding or changing scripts that support agents or skills.

## Workflow

1. Prefer self-contained Node ES modules with no dependency churn unless a parser is truly needed.
2. Provide `--help` for every script.
3. Support `--json` for validation and reporting scripts.
4. Write parseable data to stdout and diagnostics to stderr.
5. Use noninteractive inputs only: flags, environment variables, stdin, or files.
6. Make validation scripts idempotent and safe to retry.
7. Use distinct nonzero exit codes only when callers need to branch by failure type.
8. Run the script in audit mode before using strict gates.

## Sources

- Agent Skills script guidance recommends noninteractive scripts, helpful errors, structured output, idempotency, dry-run support, and predictable output size.