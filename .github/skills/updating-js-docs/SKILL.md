---
name: updating-js-docs
description: 'Use when: updating JSDoc/TSDoc, API examples, generated README source inputs, conceptual docs, or public documentation comments.'
argument-hint: 'Describe the API or docs surface, source files to edit, example or citation needs, and docs generation decision.'
user-invocable: false
disable-model-invocation: false
---

# Updating JS Docs

Use this skill to improve source documentation that feeds public docs.

Rules:
- Keep public docs atemporal and free of internal plan language.
- Prefer short examples that compile against the current public API.
- Add Mermaid diagrams or citations only when they materially improve understanding.
- Run or recommend `npm run docs` when generated outputs should refresh.

Return docs changed, generated-output decision, validation command, and residual doc gaps.