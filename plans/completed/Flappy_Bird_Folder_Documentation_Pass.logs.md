# Flappy Bird Folder Documentation Pass Log

**Status:** [DONE]

## Audit scope

- Objective: raise the educational quality of the Flappy Bird folder README
  openings without hand-editing generated README output.
- The pass covered the root-to-leaf reading path across the example's trainer,
  simulation, worker, and browser-entry documentation surfaces.

## Durable milestones

### [DONE] Folder inventory and route closure

- Reviewed the tracked Flappy Bird README-owning folders explicitly rather than
  treating the example as a single coarse documentation surface.
- Closed the reading route from the example root through the system chapters so
  a new reader can follow training, simulation, and browser playback flow.

### [DONE] Source-owned intro remediation

- Strengthened the chapter openings that needed better source-owned framing.
- Corrected generated intro ownership with targeted `docs.order.json` controls
  where a types shelf or weak facade was producing the wrong opening.

### [DONE] Worker and browser-entry documentation alignment

- Fixed worker-boundary and browser-entry framing so the generated docs better
  explain host/runtime/message ownership instead of just listing symbols.

## Controls and evidence

- Generated README surfaces stayed read-only.
- Source-affecting passes were validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.

## Reopen triggers

- Flappy Bird chapter openings drift below the current educational-docs bar.
- A changed or new boundary needs fresh intro-owner mapping.
- Generated chapter ownership regresses and needs new ordering controls.
