# Step 0 Kickoff Guide

Use this reference when the active task is to start or expand
`plans/Population_Save_Resume_and_Checkpointing.md` before any save/load code is
written.

## Goal

Finish Step 0 with a checkpoint plan that is specific enough to support a small,
test-first implementation pass instead of another broad discovery round.

## Required deliverables

1. A state-owner matrix.
2. An exactness category matrix.
3. A strict-restore failure matrix.
4. A shortlist of the first red-phase tests for Step 2 and Step 3.
5. Synchronized tracker updates in `plans/Population_Save_Resume_and_Checkpointing.md`,
   `plans/Roadmap.md`, and `plans/README.md` when the checkpoint lane becomes
   active.

## State-owner matrix template

Use a compact table or bullet list with these fields:

| State slice      | Current owner   | Full exact resume | Light mode | Existing surface | Gap or note               |
| ---------------- | --------------- | ----------------- | ---------- | ---------------- | ------------------------- |
| generation index | checkpoint seam | required          | optional   | orchestration    | restore before evaluation |

Minimum state categories to classify:

- generation and high-level run metadata,
- population ordering and per-genome scores,
- network snapshots and graph identity,
- species state,
- RNG internal state,
- orchestration counters and innovation state,
- adaptive mutation or controller state,
- downstream metadata extension needs.

## Exactness categories

Every discovered state slice should land in exactly one of these buckets:

- Required for full exact resume.
- Allowed in light mode only.
- Optional metadata that does not alter future behavior.
- Out of scope for v1.

If a state slice changes future trajectory and is missing, it cannot remain in a
metadata bucket.

## Strict-restore matrix

For each exact-resume requirement, record the expected behavior when the field
is missing:

- strict mode: throw or fail explicitly,
- non-strict mode: downgrade to best-effort with explicit diagnostics,
- never: silently claim exact replay.

## Repo-specific gotchas

- Reuse the existing network serialization boundary for graph state; Step 0 is
  not a reason to invent a second network serializer.
- Seed is not state. Exact resume needs the generator's current internal state.
- Light mode is a restart convenience contract, not an exact replay contract.
- Worker pools, transport assets, and predictor channels are adjacent runtime
  concerns, not checkpoint-owned state in v1.
- Leave room for downstream metadata extensions such as NEATchat without
  allowing those extensions to redefine the base checkpoint contract.

## Validation loop

1. Read `plans/README.md`, `plans/Roadmap.md`, and
   `plans/Population_Save_Resume_and_Checkpointing.md`.
2. Cross-check the draft against `references/checkpoint-sources.md`.
3. Confirm the active frontier is small enough for a single red-phase test.
4. Update the plan and roadmap/index together if the lane becomes `[WIP]`.
5. Stop before implementation if the save/load entry surface is still unclear.
