# Utils Docs Plan Log

**Status:** [DONE]

## Audit scope

- Objective: improve the generated utility chapter openings, especially around
  memory heuristics and snapshot-layer mechanics, while keeping the lane
  behavior-neutral.

## Durable milestones

### [DONE] Utility opening upgrades

- Strengthened the `src/utils/memory.ts` and `src/utils/memory.utils.ts`
  openings so they explain heuristic rationale and aggregation behavior more
  clearly.

### [DONE] Diagram support in utility chapters

- Added Mermaid-backed visuals where they improved the explanation of snapshot
  layers and memory behavior.

### [DONE] Docs generation unblock

- Fixed a pre-existing Mermaid rendering failure outside the utility modules so
  global docs generation could complete reliably again.

## Controls and evidence

- Validation used `npm run docs` and `npx tsc --noEmit -p tsconfig.json`.
- The pass did not require runtime behavior changes.

## Reopen triggers

- Utility chapter openings drift or new utility surfaces are added.
- Mermaid or generator behavior causes docs regression in this lane.
