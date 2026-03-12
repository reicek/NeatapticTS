# Split Workflow Checklist

Use this checklist during each SOLID split session.

## Before Editing

1. Confirm the split root and the single durable step for this session.
2. Read the nearest folder `README.md` first.
3. Read the nearest useful parent `README.md` if the boundary spans sibling
   areas.
4. Read `plans/README.md`.
5. Read only the most relevant plan file, with at most one additional related
   plan if needed.
6. Inspect the current public surface, its closest helpers, and its main
   consumers.
7. Decide whether the current file should remain a public facade, a
   compatibility shim, or the true orchestration entrypoint.

## During the Split

1. Keep exactly one active todo item.
2. Move one responsibility cluster at a time.
3. Prefer a dedicated subfolder when the file is a real subsystem.
4. Keep the main entrypoint orchestration-first.
5. Preserve stable imports where possible.
6. Improve JSDoc on touched public surfaces while splitting.
7. Add narrow tests only when they increase confidence in the extracted
   boundary.

## Documentation Pass

1. Re-read the generated README mentally through the source JSDoc you touched.
2. Add conceptual “what/why” text where the README would otherwise be too dry.
3. Add short examples for exported APIs with non-obvious behavior.
4. Mention defaults, invariants, error cases, and performance notes when they
   matter.
5. Suggest high-value background reading in JSDoc prose for important concepts
   when that would help users learn from the API.

## Validation

1. Run file diagnostics for touched files.
2. Run focused tests if the extracted logic has meaningful behavior to lock in.
3. Run `npm run docs` when JSDoc or folder shape changed.
4. Update the plan immediately after the step is complete.
5. Stop and produce the next-session handoff prompt.