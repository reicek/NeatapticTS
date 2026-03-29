# Render Docs HTML SOLID Split

**Status:** [DONE]

## Scope

- Consolidate the HTML docs renderer into a stable folder-owned boundary.
- Improve link correctness and sidebar behavior while keeping docs rendering
  behavior stable.

## Final state

- The HTML renderer now uses stable chapter ownership for assets, Mermaid,
  navigation, pages, shared logic, and types.
- Link rewriting and sidebar behavior were corrected as part of the split
  follow-through.
- No active backlog remains; this file is now a reopen point for future docs
  renderer work.

## Audit summary

- Validation used file diagnostics and `npm run docs`.
- The renderer root entrypoint remained stable through the split.

## Reopen conditions

- Future renderer work makes the current folder structure insufficient.
- Docs-site link rewriting or sidebar behavior regresses.
- Mermaid or navigation ownership needs another split.

## Audit log

- Durable completion notes now live in
  [render-docs-html-solid-split.logs.md](render-docs-html-solid-split.logs.md).
