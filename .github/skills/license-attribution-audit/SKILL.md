---
name: license-attribution-audit
description: 'Audit source references and license notes for NeatapticTS workflow customizations. Use when external standards such as Agent Skills, VS Code docs, OpenSpec, or Superpowers inform agent, skill, plan, script, or documentation changes.'
argument-hint: 'List the external sources used, target files, whether text was summarized or copied, and required license notes.'
user-invocable: false
disable-model-invocation: false
---

# License Attribution Audit

Use this skill whenever external workflow guidance informs repository customizations.

## Workflow

1. Identify each external source and its license or documentation terms.
2. Summarize patterns in original words instead of copying large passages.
3. Put attribution in internal plans, skills, or references where the source informs durable workflow.
4. Do not add third-party license headers to project files unless the user explicitly requests it.
5. Record unknown license details as blockers before implementation.

## Known Sources

- Agent Skills repository: code Apache-2.0, documentation CC-BY-4.0.
- Fission-AI OpenSpec: MIT.
- obra Superpowers: MIT.
- VS Code documentation: cite as Microsoft/VS Code documentation source for supported fields and behavior.