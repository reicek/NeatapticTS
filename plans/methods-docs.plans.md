# Methods Docs Plan

**Status:** [DONE]

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
- [src/methods/connection/connection.ts](../src/methods/connection/connection.ts)
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

### Connection chapter pass

Goals:
- Make the connection family read like a wiring-policy chooser instead of a
  terse list of topology names.
- Clarify when dense connectivity, dense-without-self-links, and one-to-one
  alignment are the right structural defaults.

Progress:
- Reframed [src/methods/connection/connection.ts](../src/methods/connection/connection.ts)
  around the structural question the family answers, with stronger source-first
  guidance for the three built-in wiring patterns.
- Tightened each connection policy entry so the generated chapter explains the
  wiring bias each option introduces and includes a minimal example for quick
  recall.

Remaining gaps:
- The thin structural families are now better aligned, so future work here is
  more likely to be polish than rescue.
- If the methods root chapter gets another pass later, its short structural
  summaries for both gating and connection can be harmonized again for tone.

Next step:
- Re-read the methods root chapter and decide whether the next highest-value
  pass is a small root-summary alignment pass or a deeper polish pass on one of
  the larger method families.

### Methods root alignment pass

Goals:
- Re-align the root methods introduction so its structural summary matches the
  stronger gating and connection chapters.
- Keep the top-level chapter compact while making the difference between
  routing control and wiring layout explicit.

Progress:
- Tightened [src/methods/methods.ts](../src/methods/methods.ts) so the root
  chapter now describes `gating` and `groupConnection` as different parts of
  the structural vocabulary instead of bundling them together too loosely.
- Cleaned the duplicated top-level module JSDoc in
  [src/methods/methods.ts](../src/methods/methods.ts) so the generated root
  opening stays compact and source-first.

Remaining gaps:
- The methods root is now aligned with the thin structural chapters, so the
  next useful pass is more likely to be selective polish than root framing.
- Larger families such as mutation or rate may still benefit from later
  refinement, but they no longer block the root chapter from reading clearly.

Next step:
- Choose the next highest-value polish pass among the larger method families,
  or stop here if the methods shelf is sufficiently aligned for now.

### Rate and mutation polish pass

Goals:
- Smooth the generated docs for two larger method-adjacent chapters without
  reopening their runtime behavior.
- Remove remaining generated-doc rough edges such as duplicated framing and
  awkward parameter prose.

Progress:
- Tightened [src/methods/rate/rate.ts](../src/methods/rate/rate.ts) so the
  class-level chapter no longer repeats the full module opening and instead
  acts like a practical chooser for schedule builders.
- Filled the missing return and option guidance for the reactive and warm
  restart rate helpers in
  [src/methods/rate/rate.ts](../src/methods/rate/rate.ts) so the generated API
  shelf reads more evenly.
- Polished [src/neat/mutation/mutation.ts](../src/neat/mutation/mutation.ts)
  with a clearer root reading order and cleaner parameter descriptions so the
  generated mutation chapter reads less like raw annotation output.

Decision:
- Stop the methods documentation lane here for now; the root and the highest-
  leverage family chapters now read coherently enough that further work is
  polish, not structural rescue.

Next step:
- Pause this plan and revisit only if a later docs pass exposes a specific
  regression, stale generated wording, or a newly expanded method family.