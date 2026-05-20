---
name: subagent-delegation-patterns
description: 'Prepare precise subagent task packets for NeatapticTS. Use when deciding between sequential and parallel specialist calls, using runSubagent, assigning hidden agents, limiting tool scope, or validating specialist output contracts.'
argument-hint: 'Describe the parent phase, specialist needed, task packet, expected output format, and whether calls can run in parallel.'
user-invocable: false
disable-model-invocation: false
---

# Subagent Delegation Patterns

Use this skill when a phase agent needs specialist help.

## Workflow

1. Delegate only when context isolation or a narrower tool set improves the result.
2. Keep each specialist task packet self-contained: goal, files, constraints, output format, and no-edit/read-only limits.
3. Use parallel subagents only for independent read-only discovery.
4. Use sequential subagents when later work depends on earlier evidence or plan state.
5. Require specialists to return compact evidence and next-step recommendations, not full workflows owned by skills.
6. Record durable decisions in `plans/Agentic_Workflow_Architecture.plans.md`.

## Specialist Packet Template

```text
Role: <hidden specialist>
Task: <one narrow objective>
Files or plans: <bounded list>
Constraints: <read-only/edit/validation limits>
Return: <exact output fields>
```

## Sources

- VS Code custom agents documentation supports custom agents as subagents with restricted `agents` allow-lists.
- Agent Skills best practices favor coherent units and progressive disclosure.