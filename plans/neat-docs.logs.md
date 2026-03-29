# NEAT Root Docs Plan Log

**Status:** [DONE]

## Audit scope

- Objective: improve the generated NEAT root and root-facing chapter openings so
  the public surface teaches orchestration and concepts before raw symbols.

## Durable milestones

### [DONE] Root NEAT controller framing

- Strengthened the root NEAT controller documentation so the generated root
  surface explains what the NEAT orchestration layer does and how readers
  should navigate the surrounding chapter set.

### [DONE] Root-facing chapter upgrades

- Upgraded the RNG and lineage openings and related root-facing chapters so
  they read as conceptual introductions rather than thin compatibility shelves.

### [DONE] Types/defaults and lineage follow-through

- Split or redirected the root-facing types/defaults surfaces where needed and
  expanded lineage with stronger diagrams, examples, and references.

## Controls and evidence

- Documentation-affecting work was validated with `npm run docs` and
  `npx tsc --noEmit -p tsconfig.json`.
- The final result is a stable reopen baseline for future NEAT docs drift.

## Reopen triggers

- Root-facing NEAT chapters regress after future API or structure work.
- Generator ordering produces weak opening ownership again.
- Another public NEAT surface needs chapter-map treatment.
