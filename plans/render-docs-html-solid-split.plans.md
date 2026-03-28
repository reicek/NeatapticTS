# Render Docs HTML SOLID Split

**Status:** [DONE]

## Scope

- Consolidate the HTML docs renderer into a folder-owned boundary rooted at `scripts/render-docs-html/`.
- Keep `scripts/render-docs-html.ts` as the stable orchestration entrypoint.
- Remove the flat `scripts/render-docs-html.sidebar.ts` file once the new direct-path chapter imports are in place.
- Improve link correctness, sidebar clarity, and docs UX where those issues are exposed by the split.

## Current state

- Split root: `scripts/render-docs-html/` with `scripts/render-docs-html.ts` as the stable orchestration entrypoint
- README inventory for `scripts/`: none under `scripts/**/README.md`
- Relevant plan: [plans/Interactive_Examples_and_Learning_Path.md](../plans/Interactive_Examples_and_Learning_Path.md)
- Relevant repo memory: `/memories/repo/scripts_docs_sidebar_boundary.md`
- Stable entrypoint: `scripts/render-docs-html.ts`
- Landed chapter files:
  - `render-docs-html.assets.ts`
  - `render-docs-html.mermaid.ts`
  - `render-docs-html.navigation.ts`
  - `render-docs-html.pages.ts`
  - `render-docs-html.shared.ts`
  - `render-docs-html.types.ts`
- Closed issues in this step:
  - nested docs content links now resolve through compiled docs targets or repository URLs instead of remaining raw repo-relative hrefs,
  - deep sidebar labels now use clearer formatted names instead of collapsing to ambiguous leaf-only segments,
  - examples-first showcase now collapses to a lighter compact mode away from onboarding/example surfaces.

## Coverage backlog

### [DONE] Consolidated renderer folder split

- Created `scripts/render-docs-html/` with focused chapters for shared helpers, navigation, Mermaid handling, page rendering, static assets, and shared contracts.
- Kept `scripts/render-docs-html.ts` orchestration-first and moved the heavy logic behind direct chapter imports.
- Deleted `scripts/render-docs-html.sidebar.ts` after the direct-path migration landed.
- Folded link-rewrite fixes and targeted sidebar UX improvements into the same step.

### [DONE] Educational docs follow-up

- Completed the required `educational-docs` follow-up on the touched renderer boundary.
- Strengthened chapter-level documentation and shared type descriptions so the renderer reads as an intentional docs-tooling subsystem rather than a loose helper shelf.

## Validation

- File diagnostics: clean for `scripts/render-docs-html.ts` and all files under `scripts/render-docs-html/`.
- `npm run docs`: passed.

## Immediate next steps

- No further split work is queued for this boundary right now.
- Future work should start from `scripts/render-docs-html/` and preserve `scripts/render-docs-html.ts` as the stable entrypoint.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Use solid-split for #file:render-docs-html.ts and #file:render-docs-html.sidebar.ts.
Plan: plans/render-docs-html-solid-split.plans.md.
Current boundary: scripts/render-docs-html.
Completed context: scripts has no local README surface, the stable entrypoint remains scripts/render-docs-html.ts, the old flat sidebar file is deleted, and the folder boundary now owns assets, Mermaid handling, navigation, shared helpers, page rendering, and shared contracts.
If this boundary reopens: continue from the folderized renderer, preserving the improved content-link rewriting and the lighter examples/sidebar behavior instead of recreating flat helper files.
Repo standard: direct-path migration with no compatibility shims by default, and the small-chapter README pattern applies to scripts/tooling boundaries too.
Required validations: file diagnostics for touched files, then npm run docs.
Worktree caution: npm run docs rewrites generated docs outputs and there may already be unrelated generated-file drift in the worktree.
```
