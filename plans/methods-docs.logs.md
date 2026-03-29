# Methods Docs Plan Log

**Status:** [DONE]

## Audit scope

- Objective: turn the root-facing `src/methods/` chapters into educational
  introductions that explain the policy choices behind each family.

## Durable milestones

### [DONE] Root methods chapter framing

- Strengthened the root methods chapter so it introduces the family map and the
  role each method family plays in the library.

### [DONE] Family-level educational reframing

- Reframed activation around transfer-curve choice, gating around placement
  semantics, and connection around wiring policy so the generated openings
  answer reader questions instead of just listing exports.

### [DONE] Polish follow-through

- Completed the root alignment and rate/mutation polish passes needed to make
  the whole methods lane read consistently.

## Controls and evidence

- Documentation updates were validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- No runtime behavior changes were required for this pass.

## Reopen triggers

- Methods chapter openings drift back toward shelf-like generated output.
- A new family or reordered source owner needs a documentation refresh.
