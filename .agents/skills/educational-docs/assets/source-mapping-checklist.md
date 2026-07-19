# Generated README Source Mapping Checklist

Use this checklist when the visible documentation target is a generated
`README.md` and the real authoring surface is JSDoc in source files.

## Goal

Trace every important README section back to the source comments that actually
produce it, then edit those sources instead of editing the generated file.

## Workflow

1. Confirm the README is generated.
   - In this repo, folder `README.md` files under `src/` are generated.
   - Treat them as read-only outputs.
2. Read the full generated README first.
   - Mark the sections that feel weak, missing, stale, or too mechanical.
3. Build a source map.
   - Identify the module entrypoint.
   - Identify exported classes, functions, constants, and types.
   - Identify companion orchestration files that likely contribute the public
     story.
4. Search for the wording you want to improve.
   - Find existing JSDoc phrases that are surfacing verbatim.
   - Find missing exports with no useful comments.
5. Rank the likely authoring sources.
   - Main entrypoint or facade.
   - Shared constants or configuration files.
   - Public helper modules.
   - Public types that carry conceptual meaning.
6. Edit the smallest number of source files that can improve the generated
   surface coherently.
7. Regenerate docs.
   - Run `npm run docs`.
8. Compare intent versus output.
   - Did the new README actually become easier to read?
   - Did the introduction gain purpose?
   - Did section ordering still make sense?
9. Tighten the source comments again if needed.

## Heuristics For Finding The Real Source

- If the README introduction is thin, the top-level module or facade usually
  needs stronger JSDoc.
- If constant names appear with no meaning, the constants file usually needs
  educational JSDoc.
- If the README lists helpers without context, exported helper functions or
  types likely need clearer descriptions.
- If the README lacks a sense of flow, the orchestration entrypoint usually
  needs a better overview comment or examples.

## Verification Questions

Ask these after regeneration:

1. Would a new reader understand what the module is for?
2. Can they tell which file is the public surface?
3. Are important invariants and defaults visible?
4. Does the README feel intentionally authored, not accidentally emitted?
5. Can the reader tell what to open next?
