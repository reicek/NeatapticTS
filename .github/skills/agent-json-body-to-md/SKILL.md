---
name: agent-json-body-to-md
description: 'Use when: converting the JSON body of a .github/agents/*.agent.md file to Markdown, replacing the raw JSON block with the formatted MD output, or running scripts/agent-customization/json-to-md.mjs against an agent description body.'
argument-hint: 'Name the target agent file (relative to .github/agents/), and confirm whether the YAML frontmatter should remain untouched.'
user-invocable: false
disable-model-invocation: false
---

# Agent JSON Body to MD

This skill converts the JSON body of a NeatapticTS `.github/agents/*.agent.md` file into Markdown and rewrites the file so that the body is human-readable Markdown while the YAML frontmatter (between the leading and trailing `---` fences) is preserved unchanged.

It exists because many agent descriptions ship as a single raw JSON object (with `mission`, `constraints`, `default_flow`, `tracker_recovery`, `if_blocked`, `output_contract`, etc.) and that body is hard to skim, review, and diff. The conversion uses the existing in-repo tool `scripts/agent-customization/json-to-md.mjs` so the output style stays consistent with the rest of the agent system.

## When to Use

- A `.agent.md` body is still raw JSON and should be converted to readable Markdown.
- A new SDLC orchestrator (e.g. `00-helping`, `01-planning`, … `07-logging`) is being added and its body needs the same Markdown treatment as sibling agents.
- An existing agent is being rewritten and the body should be re-emitted as Markdown for reviewability.
- The conversion needs to be re-run after the JSON body changes (e.g. constraints, flow steps, or recovery rules were updated).

## Task Packet

Include the target file, the desired scope, and whether the script output should be written back to the file in-place.

```text
Use agent-json-body-to-md for <relative-path-under-.github/agents/>.
Body scope: <full body | specific top-level keys>
Rewrite in-place: <yes | no — print to stdout only>
Preserve frontmatter: <yes>
```

## Tooling Overview

- Script: `scripts/agent-customization/json-to-md.mjs`
- Export: `jsonToMarkdown(obj)` — pure function, no side effects, no filesystem access.
- Input: a plain JavaScript object whose top-level keys become `## Title Case` sections.
- Output: a Markdown string. Arrays become bullet lists, strings are returned as-is, nested objects are recursively formatted with bold keys.
- The script is invoked from a one-shot Node `node --input-type=module -e "..."` runner; do not edit the script itself to read a file or write a file — keep conversion pure and do the file rewrite with the internal `edit()` tool.

## Required Workflow

1. Read the target `.github/agents/<name>.agent.md` file in full with `view()` so the frontmatter boundaries and the raw JSON body are both visible.
2. Confirm the file actually has a YAML frontmatter block (a leading `---` line) and a JSON body section. If the file is already pure Markdown or has no JSON body, stop and report `TASK_STATUS: PARTIAL` with the reason instead of fabricating a conversion.
3. Capture the exact byte range of the YAML frontmatter — everything from the first `---` line through the closing `---` line, inclusive — so it can be re-emitted unchanged. Do not reformat, reindent, or re-quote the frontmatter; the only field that should be touched is the JSON body.
4. Parse the JSON body in memory (do not write a temp file). If the JSON does not parse, stop and report the parse error; do not attempt partial repair.
5. Run the converter via Node:
   ```bash
   node --input-type=module -e "import('./scripts/agent-customization/json-to-md.mjs').then(m => { console.log(m.jsonToMarkdown(<obj>)); });"
   ```
   where `<obj>` is the parsed body. Capture stdout as the new Markdown body.
6. Assemble the replacement file:
   - Line 1: original frontmatter block (unchanged, including the closing `---`).
   - Blank line, then the captured `jsonToMarkdown(...)` output verbatim, then a blank line, then any trailing non-JSON content that originally followed the JSON block (e.g. the fenced ` ```structured-v1 … ``` ` block that some agents use for their output contract).
7. Use the internal `edit()` tool — not the `create()` tool — to perform the substitution. `edit()` is the safe path: it preserves every line that does not need to change and produces a single, reviewable diff. `create()` would risk overwriting unrelated content or losing trailing fences.
   - Pass `old_str` as the entire original body (frontmatter + JSON block) and `new_str` as the new frontmatter + Markdown body.
   - Keep trailing fences (e.g. the ` ```structured-v1 ` output contract block) outside the Markdown converter so they stay syntactically intact and continue to render as a fenced code block.
8. After the edit, re-read the file with `view()` and confirm:
   - The frontmatter block is byte-identical to the original (only the body changed).
   - The body is now Markdown, not JSON.
   - Any trailing fenced blocks are preserved.
9. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` (or `--strict` when the migration target requires it) to confirm the frontmatter still validates. Frontmatter-only validators are expected to keep passing because the body is treated as opaque Markdown.
10. Report `FILES_CHANGED`, `ACTIONS_TAKEN`, and `VALIDATION_EVIDENCE` in the structured-v1 output contract.

## Worked Example: 01-planning.agent.md

1. `view(".github/agents/01-planning.agent.md")` — observe:
   - Lines 1–17 are the YAML frontmatter.
   - Line 18 is blank.
   - Lines 19–61 are the raw JSON body (object with `mission`, `constraints`, `default_flow`, `tracker_recovery`, `if_blocked`, `output_contract`).
   - Lines 63–96 are a trailing fenced ```` ```structured-v1 ```` output contract block that must stay untouched.
2. Capture the frontmatter verbatim (lines 1–17 inclusive of the closing `---`).
3. Parse lines 19–61 as JSON. The script does not need a temp file; embed the object literal inline in the Node `-e` invocation.
4. Run:
   ```bash
   node --input-type=module -e \
     "import('./scripts/agent-customization/json-to-md.mjs').then(m => { const obj = <inline-obj>; console.log(m.jsonToMarkdown(obj)); });"
   ```
   Capture stdout.
5. Use `edit()` with `old_str` = the entire original file (frontmatter + JSON body) and `new_str` = the original frontmatter + a blank line + the Markdown body + a blank line + the trailing ` ```structured-v1 ` block. Keep the trailing block exactly as it was.
6. Re-read the file; confirm the body now contains `## Mission`, `## Constraints`, `## Default Flow`, `## Tracker Recovery`, `## If Blocked`, `## Output Contract` headings, each followed by the appropriate bullet list or paragraph from the JSON, and that the ` ```structured-v1 ` fence is still present at the end.
7. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` to confirm the frontmatter is still valid.

## Guardrails

- Do not hand-edit `scripts/agent-customization/json-to-md.mjs` to read or write files. Keep the converter pure; do the file rewrite with the `edit()` tool.
- Do not use `create()` to overwrite the agent file in place. Always use `edit()` with a precise `old_str` / `new_str` pair so the diff is reviewable and no unrelated content is lost.
- Do not include the trailing ` ```structured-v1 ` (or any other fenced block that follows the JSON body) inside the JSON object passed to `jsonToMarkdown`. Fenced code blocks must stay fenced; running them through the converter would either escape their backticks or render them as inline text.
- Do not reformat the YAML frontmatter as part of this skill. The only field that should change is the body. Any frontmatter change belongs to `updating-agent-frontmatter` or `agent-frontmatter-standards`.
- Do not delete the trailing newline at the end of the Markdown body. Markdown renderers expect a trailing newline and most diff tools add one automatically; do not strip it.
- Do not run `jsonToMarkdown` against an array or a scalar. It expects a top-level object; passing a JSON array will produce a single `## 0` section and silently corrupt the output.
- Do not silently drop keys that the converter does not know how to render. `jsonToMarkdown` handles `string`, `number`, `Array`, and nested `object`; if a future key uses a non-JSON type, stop and report it instead of writing a partial body.
- Do not proceed if the JSON body does not parse; surface the parse error and let the caller decide whether to repair the JSON first.
- Do not skip the post-edit `view()` re-read. The re-read is the only check that proves the frontmatter is byte-identical and the trailing fences are still present.

## Expected Final Output

- The target `.github/agents/<name>.agent.md` has a YAML frontmatter that is byte-identical to the original and a body that is human-readable Markdown produced by `scripts/agent-customization/json-to-md.mjs`.
- Any trailing fenced blocks (e.g. ` ```structured-v1 `) are preserved as fenced code blocks, not flattened into the Markdown body.
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` exits with no errors.
- The structured-v1 output contract reports `FILES_CHANGED`, `ACTIONS_TAKEN`, and `VALIDATION_EVIDENCE` with the exact commands and exit codes.
