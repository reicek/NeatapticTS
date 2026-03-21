# Methods Docs Plan

**Status:** [WIP]

## Scope

This plan tracks educational-docs passes for the shared method vocabulary under
[src/methods/README.md](../src/methods/README.md).

Primary reader:
- readers who want the reusable training and evolutionary method families to
  read like a guided policy shelf rather than a raw export list.

Primary surfaces:
- [src/methods/methods.ts](../src/methods/methods.ts)
- [src/methods/cost/cost.ts](../src/methods/cost/cost.ts)
- [src/methods/rate/rate.ts](../src/methods/rate/rate.ts)
- [src/methods/selection/selection.ts](../src/methods/selection/selection.ts)
- [src/methods/gating/gating.ts](../src/methods/gating/gating.ts)
- [src/methods/README.md](../src/methods/README.md)

## Session Log

### Methods root chapter pass

Goals:
- Make the generated [src/methods/README.md](../src/methods/README.md) open as
  a guided chapter instead of a raw export list.

Progress:
- Added a barrel-led introduction in
  [src/methods/methods.ts](../src/methods/methods.ts) so the chapter now opens
  with the shared method-family map.
- Added [src/methods/docs.order.json](../src/methods/docs.order.json) to keep
  `methods.ts` as the intro source and hide the barrel file section.
- Strengthened the cost, rate, and selection family surfaces with clearer
  framing and diagram-justified guidance.

Decision:
- Keep future methods work proportional and continue one family chapter at a
  time rather than reopening the entire folder.

### Activation chapter pass

Goals:
- Make the activation family read more like a chooser and less like a flat
  list of transfer-curve helpers.
- Tighten the older symbol-level prose in the utility shelf without changing
  runtime behavior.

Progress:
- Added a compact family map in
  [src/methods/activation/activation.ts](../src/methods/activation/activation.ts)
  so the generated activation chapter shows the main activation clusters before
  readers fall into the long symbol shelf.
- Strengthened
  [src/methods/activation/activation.utils.ts](../src/methods/activation/activation.utils.ts)
  with a clearer reading order and more explanatory JSDoc for the previously
  terse helper functions, especially the historical, localized, and niche
  transforms.

Decision:
- Leave the deeper per-symbol registry cleanup for a later polish pass and keep
  moving through the thin structural chapters one family at a time.

### Gating chapter pass

Goals:
- Make the gating family read like a structural chooser instead of a short
  constant shelf.
- Clarify how `INPUT`, `OUTPUT`, and `SELF` differ so readers can pick a gate
  position by intent instead of memorizing names.

Progress:
- Reframed [src/methods/gating/gating.ts](../src/methods/gating/gating.ts)
  around the placement question the family answers, with stronger source-first
  chooser guidance for the three gate positions.
- Tightened each gate entry so the generated chapter explains the control
  surface each option creates and includes a minimal example for quick recall.

Remaining gaps:
- The sibling structural family for connection policies is still thinner than
  the rest of the methods shelf and likely needs the same chapter-first framing.
- If the methods root chapter gets another pass later, its short gating summary
  can be aligned again with any wording refinements that come out of the
  connection pass.

Next step:
- Move to the connection family next and give its structural wiring vocabulary
  the same chooser-first treatment used for gating.