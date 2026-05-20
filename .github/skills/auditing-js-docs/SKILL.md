---
name: auditing-js-docs
description: 'Use when: auditing JSDoc/TSDoc quality, generated README inputs, missing examples, stale public API docs, or educational documentation gaps.'
user-invocable: false
disable-model-invocation: false
---

# Auditing JS Docs

Use this skill to find documentation gaps before editing docs.

Rules:
- Read generated READMEs as context, then verify the source JSDoc.
- Do not hand-edit generated `src/**/README.md` files.
- Flag missing examples, unclear semantics, stale invariants, and missing citations.
- Recommend `educational-docs` when the update needs diagrams or deeper source mapping.

Return surfaces audited, gaps found, source files to edit, and docs validation needed.