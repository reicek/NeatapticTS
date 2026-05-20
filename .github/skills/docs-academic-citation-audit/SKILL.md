---
name: docs-academic-citation-audit
description: 'Audit NeatapticTS educational documentation quality for customization or source docs. Use when adding JSDoc, Mermaid diagrams, academic references, generated README inputs, source attributions, or checking that documentation meets research-and-education standards.'
argument-hint: 'Describe the documentation surface, source files, generated outputs, citations needed, and validation command.'
user-invocable: false
disable-model-invocation: false
---

# Docs Academic Citation Audit

Use this skill when documentation changes need educational or academic quality.

## Workflow

1. Keep public docs atemporal: concepts, invariants, tradeoffs, and examples instead of roadmap history.
2. Prefer source JSDoc over generated README edits for `src/**` documentation.
3. Add Mermaid diagrams when they teach faster than prose.
4. Cite academic papers or external references when explaining named algorithms or borrowed workflow standards.
5. Keep examples short, dependency-light, and aligned to public APIs.
6. Run `npm run docs` when source JSDoc or generated example inputs change.
7. Verify generated output rather than patching generated files directly.

## Coordination

- Use `educational-docs` for broad documentation passes.
- Use `license-attribution-audit` when external workflow sources inform customization docs.