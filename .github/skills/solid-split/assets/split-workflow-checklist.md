# Split Workflow Checklist

Use this checklist during each SOLID split session.

## Before Editing

1. Confirm the split root and the single durable step for this session.
2. Build a full inventory of every `README.md` under the requested root.
3. Convert that inventory into an explicit todo list so every README-owning
   folder is visible before edits begin.
4. Read the nearest folder `README.md` first.
5. Read the nearest useful parent `README.md` if the boundary spans sibling
   areas.
6. Read `plans/README.md`.
7. Read only the most relevant plan file, with at most one additional related
   plan if needed.
8. Inspect the current public surface, its closest helpers, and its main
   consumers.
9. Decide whether the current file should remain a public facade, a
   compatibility shim, or the true orchestration entrypoint.

## During the Split

1. Keep exactly one active todo item.
2. For documentation-heavy passes, keep that active item bound to one README or
   one README-owning folder at a time.
3. Move one responsibility cluster at a time.
4. Prefer a dedicated subfolder when the file is a real subsystem.
5. Keep the main entrypoint orchestration-first.
6. Preserve stable imports where possible.
7. Improve JSDoc on touched public surfaces while splitting.
8. Add narrow tests only when they increase confidence in the extracted
   boundary.

## Documentation Pass

1. Re-read the generated README mentally through the source JSDoc you touched.
2. Add conceptual “what/why” text where the README would otherwise be too dry.
3. Add short examples for exported APIs with non-obvious behavior.
4. Mention defaults, invariants, error cases, and performance notes when they
   matter.
5. If the docs need a broader tone lift, source mapping, citations, or
   Wikimedia-safe visuals, invoke `educational-docs` before regenerating docs.

## Validation

1. Run file diagnostics for touched files.
2. Run focused tests if the extracted logic has meaningful behavior to lock in.
3. Run `npm run docs` when JSDoc or folder shape changed.
4. Update the plan immediately after the step is complete.
5. Stop and produce the next-session handoff prompt.