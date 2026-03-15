# NEAT Root Docs Plan

## Scope

This plan tracks the educational-docs pass for the root NEAT public surface.
The source of truth is the generated documentation fed by [src/neat.ts](../src/neat.ts),
which renders into [src/README.md](../src/README.md) after `npm run docs`.

Primary reader:
- first-time NeatapticTS readers who want to understand how the top-level NEAT controller works before diving into chaptered internals.

Primary surfaces:
- [src/neat.ts](../src/neat.ts)
- [src/README.md](../src/README.md)

Out of scope for this pass:
- hand-editing generated READMEs under `src/`
- adding external images or citations unless a concrete teaching gap requires them
- broad rewrites of every `src/neat/**` chapter README

## Assumptions

- The generated root NEAT story is expected to live in [src/README.md](../src/README.md), not in [src/neat/README.md](../src/neat/README.md).
- The highest-leverage fix is richer JSDoc in [src/neat.ts](../src/neat.ts), especially module-level and class-level introductions.
- External sources and Wikipedia media are optional, not mandatory. If none are needed, the safest compliant outcome is to use no external media and no external citations.

## Session Log

### 2026-03-15 - Session start

Goals:
- turn the root NEAT docs into a real chapter introduction with a strong opening narrative
- add at least one diagram to orient readers around the lifecycle
- improve the public `NeatOptions` and `Neat` explanations so the generated docs teach workflow, not just signatures
- regenerate docs and verify the generated output

Progress:
- inspected [src/neat/README.md](../src/neat/README.md) and confirmed it is not the root NEAT chapter surface for this task
- traced the actual generated NEAT surface to [src/README.md](../src/README.md)
- confirmed the main gap is in [src/neat.ts](../src/neat.ts): the generated chapter currently has a bare module heading and a `default` class heading with little orientation

Next action:
- rewrite root JSDoc in [src/neat.ts](../src/neat.ts), regenerate docs, and then update this plan with the verification result

### 2026-03-15 - Session end

Achievements:
- rewrote the root module introduction in [src/neat.ts](../src/neat.ts) so the generated docs explain the NEAT lifecycle, the reason the controller stays orchestration-first, and where to read next
- added a Mermaid lifecycle diagram in [src/neat.ts](../src/neat.ts) to orient readers before they hit the API surface
- promoted the public class docs from an anonymous generated `default` section to a named `Neat` section by switching to a named class with a default export in [src/neat.ts](../src/neat.ts)
- expanded the public teaching surface for `NeatOptions`, constructor semantics, evolution, evaluation, telemetry, objectives, RNG state, and persistence helpers in [src/neat.ts](../src/neat.ts)
- regenerated [src/README.md](../src/README.md) with `npm run docs` and verified that:
	- the opening now presents a long NEAT-oriented chapter introduction
	- the Mermaid lifecycle diagram renders into the generated README markdown
	- the public class heading now renders as `Neat` instead of `default`

Validation:
- `npm run docs` completed successfully
- no editor diagnostics were reported for [src/neat.ts](../src/neat.ts) or [plans/nead-docs.plans.md](./nead-docs.plans.md)

External sources and media:
- none added in this session
- no Wikipedia or third-party media was used, so no extra attribution or license obligations were introduced

Remaining gaps:
- the generated `src/README.md` opening is now NEAT-centered because the docs generator lifts the module intro high in the root file; that is acceptable for this pass, but a future docs-generator refinement could place per-file introductions closer to their own file sections
- several lower-level public methods in [src/neat.ts](../src/neat.ts) still have shorter descriptions than the best chaptered READMEs under `src/neat/**`

Next step:
- if a follow-up session happens, deepen the most reader-visible method docs in [src/neat.ts](../src/neat.ts) for export/import, species history, and multi-objective inspection so the generated reference continues to read like a guided API tour

## Handoff Prompt

```text
Continue the educational-docs pass for the root NEAT surface using plans/nead-docs.plans.md as the source of truth.

Current target:
- src/neat.ts feeding src/README.md

What to verify next:
- skim the generated src/README.md opening to confirm the wording still fits the broader src surface after future doc-generator changes
- decide whether the next docs pass should deepen root Neat method explanations or move into one adjacent chapter such as telemetry or multi-objective

If more work is needed:
- keep edits source-first in src/neat.ts
- do not hand-edit generated src/**/README.md files
- update plans/nead-docs.plans.md with achievements, remaining gaps, and the next concrete step before ending the session
```
