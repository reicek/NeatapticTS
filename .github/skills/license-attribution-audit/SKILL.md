---
name: license-attribution-audit
description: 'Audit source references and license notes for NeatapticTS workflow customizations. Use when external standards such as Agent Skills, VS Code docs, OpenSpec, or Superpowers inform agent, skill, plan, script, or documentation changes.'
argument-hint: 'List the external sources used, target files, whether text was summarized or copied, and required license notes.'
user-invocable: false
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# License Attribution Audit

Use this skill whenever external workflow guidance informs repository
customizations. Attribution keeps the project honest about what is original
and what derives from published external standards.

This skill owns the workflow for identifying external source licenses, ensuring
text is handled according to those licenses, placing attribution in the correct
internal location, and recording unknown license details as blockers.

## When to Use

- A new skill, agent, or script draws on wording or patterns from an external
  specification such as Agent Skills, VS Code documentation, OpenSpec, or
  Superpowers.
- An existing skill or plan references an external source and the attribution
  is missing or incomplete.
- A documentation pass quotes or paraphrases a third-party workflow standard
  and the source has not been cited.
- A references file under `.github/skills/*/references/` is being created or
  updated and the source license needs to be documented.
- An external source with an unknown license was used and the project needs a
  recorded blocker before the change can be merged.

## Task Packet

Pass a compact packet listing the external sources, where they were used, and
what attribution action is needed.

```text
Use license-attribution-audit for new worker-inference-transport references.
External sources: Agent Skills best practices (Apache-2.0 code, CC-BY-4.0 docs), VS Code worker documentation.
Used in: .github/skills/worker-inference-transport/references/worker-transport-sources.md.
Action: confirm text is summarized (not copied), add attribution in the references file.
```

## Required Workflow

1. Identify each external source used in the current change and its license or
   documentation terms.
2. Summarize patterns in original words instead of copying large passages. Do
   not reproduce verbatim excerpts from CC-licensed or proprietary docs beyond
   brief quotation.
3. Put attribution in internal plans, skills, or references files where the
   source informs durable workflow. Do not scatter attribution into generated
   README outputs.
4. Do not add third-party license headers to project source files unless the
   user explicitly requests it; attribution in references files is sufficient
   for workflow customizations.
5. Record unknown license details as blockers before implementation continues;
   do not assume a permissive license for undocumented external sources.

## Known Sources

- Agent Skills repository: code Apache-2.0, documentation CC-BY-4.0.
- Fission-AI OpenSpec: MIT.
- obra Superpowers: MIT.
- VS Code documentation: cite as Microsoft/VS Code documentation source for
  supported fields and behavior; no code reproduction needed for API shape
  descriptions.

## Guardrails

- Do not reproduce large verbatim passages from CC-BY-4.0 or proprietary
  sources; summarize instead.
- Do not merge a change that relies on an external source with an unknown
  license; record the blocker first.
- Do not add third-party license headers to project files without an explicit
  user instruction.
- Do not place attribution only in commit messages; durable attribution belongs
  in the relevant references file or plan.

## Expected Final Output

A strong attribution audit pass should produce:

- a list of every external source used, its license, and where it was
  referenced,
- confirmation that all text is summarized rather than copied where the license
  requires it,
- updated attribution in the relevant `.github/skills/*/references/` file or
  plan,
- any recorded blockers for sources with unknown license status.
