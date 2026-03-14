# Scripts Docs Sidebar SOLID Split

## Purpose

Reduce `scripts/render-docs-html.ts` as the coordination sink for docs navigation generation while keeping it as the stable docs HTML entrypoint. The immediate goal is to move sidebar-specific data and rendering behind a focused helper surface and fix the left-nav continuity contract so generated markup no longer resets the path between nested example clusters.

## Root

- Split root: `scripts`
- Nearest README reviewed: `n/a` (no `README.md` under `scripts/`)
- Parent README reviewed: `README.md`
- Relevant plan: `plans/Interactive_Examples_and_Learning_Path.md`

## Durable Rules

- Keep exactly one active step at a time.
- Update this plan immediately after each completed step.
- Preserve stable script entrypoints unless a breaking change is explicitly approved.
- Do not hand-edit generated README files; improve source JSDoc and run docs when docs surface changes.
- Keep `scripts/render-docs-html.ts` orchestration-first.

## Target Shape

- `scripts/render-docs-html.ts` remains the public docs HTML generator entrypoint.
- `scripts/render-docs-html.sidebar.ts` owns sidebar example metadata, pipe-nav rendering, grouping, and examples sidebar composition.
- Sidebar markup uses one semantic pipe-nav list per cluster instead of wrapping full pipe-nav lists inside additional list containers.
- Sidebar CSS targets the normalized pipe-nav contract rather than compensating for nested list resets.

## Steps

- [x] Step 1: Map the current boundary, stable entrypoint, and sidebar continuity seams.
- [x] Step 2: Extract the sidebar/navigation responsibility into a focused helper module and normalize examples/sidebar list markup to preserve vertical continuity.
- [x] Step 3: Tighten the remaining pipe-path visual contract now that the markup no longer nests full pipe-nav lists inside wrapper lists.
- [x] Step 4: Improve JSDoc so the touched docs-generation boundary reads naturally in generated surfaces where relevant.
- [x] Step 5: Validate the touched surface, decide whether docs regeneration is still required, and record the durable boundary note.

## Progress Notes

- Completed Step 2 by moving examples metadata, pipe-nav rendering, grouping, and sidebar assembly into `scripts/render-docs-html.sidebar.ts`.
- Normalized the examples sidebar and root examples TOC so they now mount pipe-nav lists directly instead of wrapping full `ul` fragments in extra `ul` containers.
- Narrowed sidebar CSS away from generic nested-list rules so the new markup contract can render without inherited indentation resets.
- Validation after the extraction: `npm run docs:html` passed and file diagnostics were clean for the touched renderer, sidebar helper, and theme stylesheet.
- Completed Step 3 by replacing the remaining two-row glyph stack with a box-based rail track in `scripts/render-docs-html.sidebar.ts`, so each depth column renders from stable layout boxes instead of text metrics.
- Updated `scripts/assets/theme.css` so the sidebar rail is drawn with persistent vertical columns and horizontal connectors, which keeps the left path visually continuous across the normalized example clusters.
- Step 3 validation: `npm run docs:html` passed again, file diagnostics were clean for `scripts/render-docs-html.ts`, `scripts/render-docs-html.sidebar.ts`, and `scripts/assets/theme.css`, and the generated sidebar was spot-checked in the browser.
- Completed Step 4 with a focused educational-docs follow-up in `scripts/render-docs-html.sidebar.ts`, clarifying why the helper owns the examples-first grouping and how the depth-aware rail contract uses box-based columns instead of glyph stacks.
- Step 4 validation: file diagnostics remained clean for `scripts/render-docs-html.sidebar.ts`; `npm run docs:html` was not rerun because this pass only changed source comments and did not alter emitted docs HTML.
- Completed Step 5 by validating the touched surface without widening the scope: file diagnostics were clean for `scripts/render-docs-html.ts`, `scripts/render-docs-html.sidebar.ts`, and `scripts/assets/theme.css`, and the already-generated docs page still showed the examples-first sidebar grouping with the deep-dive rail structure intact.
- Step 5 decision: `npm run docs:html` was not rerun because the post-build changes after Step 3 were documentation-only JSDoc improvements in `scripts/render-docs-html.sidebar.ts`, so there was no emitted HTML drift to refresh.
- Recorded the durable boundary note in `/memories/repo/scripts_docs_sidebar_boundary.md` so the next session can recover the split shape and validation rationale without replaying chat history.

## Done Criteria

- `scripts/render-docs-html.ts` is no longer the obvious home for sidebar-specific glyph composition.
- Sidebar markup no longer nests complete pipe-nav lists inside extra `ul` wrappers.
- The left sidebar path renders with consistent vertical continuity across example clusters.
- The docs HTML build remains green after the extraction.
- The next session can continue from this plan alone.
