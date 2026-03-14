---
name: educational-docs
description: 'Turn JSDoc and source comments into world-class educational documentation for generated README surfaces and hand-written docs. Use when a module README feels dry, a public API needs stronger explanation, or a docs pass needs Mermaid diagrams, compliant external references, or Wikimedia-safe visuals.'
argument-hint: 'Describe the target folder or public surface, the intended reader, and whether you need tone shaping, source mapping, Mermaid diagrams, citations, or Wikimedia-safe media.'
user-invocable: true
disable-model-invocation: false
---

# Educational Docs Playbook

Use this skill when documentation needs to teach, orient, and motivate rather
than merely restate types.

This repository is an educational library. The generated README output is part
of the product experience, not an afterthought. A good documentation pass should
help a first-time reader understand what the module is for, why its boundaries
exist, how the pieces fit together, and where to read next.

This skill is also the mandatory documentation follow-up for completed
`solid-split` steps unless the user explicitly opts out. When invoked from a
split workflow, treat the incoming changes as a focused boundary-sharpening
pass: explain the touched module better, keep generated surfaces in sync, and
avoid expanding into an unrelated repo-wide docs rewrite.

This file is the canonical knowledge surface for documentation work in this
repo. It owns the durable documentation workflow, tone model, source-mapping
rules, Mermaid policy, and citation/media guardrails. Companion agents such as
`Docs Scout` should stay thin: they should gather context, identify likely doc
gaps, and hand those specifics back into this skill rather than re-defining the
documentation bar themselves.

## When to Use

- A generated README feels technically correct but emotionally flat or hard to
  navigate.
- A folder under `src/` or `test/` needs richer JSDoc so doc generation becomes
  more useful.
- A split or refactor changed a module boundary and the documentation story now
  lags behind the code.
- A public API needs examples, invariants, tradeoffs, or performance notes.
- A concept would benefit from carefully cited background reading.
- A diagram or image from Wikimedia Commons could materially improve learning,
  and the license obligations can be satisfied.
- A `solid-split` step has just completed and the touched boundary now needs a
  focused educational-docs follow-up pass.

## Companion Workflow Input

When this skill is used as the follow-up step after `solid-split`, expect a
compact handoff packet rather than a broad docs request.

Preferred handoff fields:

- changed boundary, folder, or file set,
- intended reader,
- whether the public surface is generated from source JSDoc,
- the split step that just completed,
- documentation needs discovered during the split,
- whether Mermaid, citations, or Wikimedia-safe media are actually in scope,
- validation expectations such as `npm run docs`.

Compact example:

```text
Use educational-docs on the browser-entry/playback split that just finished.
Reader: first-time contributor inspecting the new playback boundary.
Generated surface: yes, source JSDoc feeds README output.
Completed split step: Step 4 - extract playback snapshot boundary.
Doc needs: tighten module purpose, explain new boundary ownership, add a small
example only if the API is non-obvious.
Mermaid: only if it materially reduces confusion.
Validate with: npm run docs
```

## Responsibility Split

Use this boundary intentionally:

- The skill owns durable documentation knowledge: workflow, tone, teaching
  standards, source mapping, visual policy, and attribution rules.
- `Docs Scout` owns read-only reconnaissance: read the nearest README surface,
  compare it with the smallest set of nearby source files, and surface likely
  JSDoc or regeneration targets.
- `Docs Scout` should not restate this skill's full documentation philosophy.
  It should pass concrete findings into this skill.
- If the skill and a companion agent drift, update the agent to follow this
  skill rather than copying the drift forward.

## Docs Scout Handoff

When `Docs Scout` is used before or alongside this skill, the useful handoff is
compact and evidence-based.

Preferred handoff fields:

- README files inspected,
- short assessment of whether the README is sufficient, stale, or dry,
- likely source-JSDoc targets,
- whether docs probably only need regeneration,
- user-facing explanation gaps,
- any recommendation to run `npm run docs`.

This handoff should narrow the doc pass. It should not replace the actual
educational-docs workflow.

## Primary Resources

- [README tone model](./assets/readme-tone-model.md)
- [Astro Bird visual style guide](./assets/visual-style-guide.md)
- [Generated README source mapping checklist](./assets/source-mapping-checklist.md)
- [Mermaid diagram playbook](./assets/mermaid-diagram-playbook.md)
- [External sources and Wikimedia media guide](./assets/external-sources-and-media.md)

## Core Promise

Use this skill to turn docs into guided tours.

The target outcome is not "more words." The target outcome is a README or API
surface that does four things well:

1. tells the reader why the module matters,
2. gives them a map before the details,
3. explains design choices and tradeoffs instead of hiding them,
4. points them toward the next useful file, concept, or experiment.

For generated folder READMEs, that map should begin immediately under the top
`#` heading. A blank title followed by file sections is not enough. If the
current source mapping or generator cannot produce a real opening, improve that
mechanism so the folder README starts with a genuine module introduction.

When a visual explanation would teach faster than prose, the target surface
should also include Mermaid Markdown diagrams or charts.

Those visuals should default to Astro Bird's neon-retro-arcade language: dark
backgrounds, blue-led structural lines, high-contrast labels, and restrained
glow on the component or path that matters most.

## Tone Model

Use the Flappy Bird example README as the reference bar for tone and structure.
Its documentation works because it does not open with file trivia. It opens with
purpose, frames practical questions, organizes the system into readable layers,
then gives the reader several ways to continue exploring.

Use Astro Bird's browser palette as the reference bar for documentation
visuals. That means dark backgrounds, cyan and electric-blue structure, limited
warm accents, and enough contrast that the docs still scan like engineering
material.

Apply that model broadly:

- lead with what the module is for,
- explain why the boundary exists,
- answer real user or reader questions,
- use named sections that create momentum,
- teach architecture and behavior together,
- finish with recommended reading or next steps when helpful.

Treat heading depth as a teaching contract, not a formatting afterthought:

- the `#` opening is the chapter introduction and should usually be the
  longest, richest prose in the document,
- the `#` opening should orient a first-time reader before file trivia or
  symbol lists appear,
- `##` sections should introduce one meaningful slice of the module with enough
  context that the reader knows why that slice exists,
- `###` sections should stay concise and specific, acting as the detail shelf
  beneath a stronger parent section,
- long runs of constants, options, or related exports should be introduced by
  an umbrella explanation before the reader hits a wall of repeated headings.

See the full rubric in [README tone model](./assets/readme-tone-model.md).
See the visual rules in [Astro Bird visual style guide](./assets/visual-style-guide.md).

## Required Workflow

1. Identify the documentation surface.
   - Decide whether the target is a generated folder README, a hand-written doc,
     or both.
2. Trace the source of truth.
   - If the README is generated, do not edit the README directly.
   - Build a source map for the exports, constants, classes, and orchestration
     files that feed the generated output.
3. Read before rewriting.
   - Read the nearest folder README, then the nearest useful parent README.
   - Read the smallest set of source files that own the public story.
4. Define the reader questions.
   - What is this for?
   - Why is it shaped this way?
   - What are the important defaults or invariants?
   - What is the shortest path to using it correctly?
5. Upgrade the source comments.
   - Add or improve JSDoc so it explains what, why, and when.
   - Add short examples where behavior is non-obvious.
   - Add performance notes, invariants, and error semantics where they matter.
6. Add concept bridges when useful.
   - If a concept is central and not obvious, add concise background reading in
     prose with a properly attributed source.
   - Prefer one high-value reference over a noisy list.
7. Evaluate Mermaid-first visuals deliberately.
  - Prefer Mermaid Markdown for architecture overviews, data flows, decision
    flows, state transitions, timelines, simple charts, and structural maps.
  - Treat GitHub README rendering as the primary Mermaid compatibility target
    unless the task explicitly says the diagram only needs to work in the
    generated HTML docs.
  - Style diagrams with the Astro Bird palette so structure remains blue-led
    and emphasis stays scarce.
  - Add diagrams when they reduce confusion, not merely because Mermaid is
    available.
  - Choose the diagram family that matches the teaching goal.
  - Validate non-trivial Mermaid syntax before finalizing when tooling is
    available.
  - Prefer conservative Mermaid syntax when the README itself is a primary
    surface on GitHub.
8. Evaluate external visuals deliberately.
  - Add images only when they improve understanding, not for decoration.
  - Prefer Wikimedia Commons files with clear free-license metadata.
  - Record author, source URL, license, and modification status.
9. Regenerate and inspect.
   - Run `npm run docs` after doc-affecting edits to generated surfaces.
   - Re-read the generated output as if you were new to the module.
10. Tighten the experience.
   - Remove repetition.
   - Replace dry restatements with explanation.
   - Make the reading order obvious.

## Post-Split Follow-Up Mode

When invoked immediately after `solid-split`, apply this narrower execution
mode:

1. Start from the changed boundary rather than from the whole subsystem.
2. Preserve the split's new ownership story and make it easier to read.
3. Prefer source-JSDoc improvements first when the README surface is generated.
4. Add diagrams only when the new boundary is still hard to understand without
  one.
5. Regenerate docs when the touched surface feeds generated README output.
6. Report the doc pass as the follow-up to the completed split step so the
  workflow remains legible.

This mode exists to keep the follow-up deterministic: every completed split
gets a documentation polish pass, but that pass stays proportional to the code
change that triggered it.

## World-Class Standards

### 1. Start with purpose, not taxonomy

A reader should understand the module's role before encountering file lists or
signature details.

For any document with a top-level `#` heading, the opening should read like the
first pages of a chapter: define the problem space, explain why this boundary
exists, and give the reader a clear path into the sections that follow.

### 2. Organize around questions, not only symbols

Good educational docs answer questions such as:

- What problem does this solve?
- Why is the architecture split this way?
- What does the runtime or algorithm assume?
- What path should I read next if I want more depth?

Match the amount of prose to the heading depth:

- put the broadest motivation and mental model under `#`,
- put section-level framing and reading order under `##`,
- keep `###` entries compact unless one specific item genuinely needs extra
  depth.

### 3. Make architecture legible

For non-trivial modules, prefer structure that gives the reader a map:

- responsibility layers,
- execution paths,
- design choices,
- reading order,
- compatibility facades versus implementation detail.

When a diagram can communicate that structure faster than prose, include one.

### 3a. Make invisible structure visible

Use Mermaid Markdown aggressively when it adds real explanatory power.

High-value uses include:

- architecture overviews,
- execution paths,
- data flows,
- decision trees,
- state machines,
- entity relationships,
- simple quantitative charts,
- prioritization views.

Keep the visual system consistent while doing so: dark canvas, blue structural
lines, readable cool-toned text, and glow reserved for the component the prose
is actively teaching.

Use the detailed chooser in [Mermaid diagram playbook](./assets/mermaid-diagram-playbook.md).

### 4. Prefer explanation over paraphrase

Avoid comments that merely rename the code in sentence form. Each public-facing
comment should add at least one of these:

- intent,
- tradeoff,
- invariant,
- failure mode,
- mental model,
- example,
- historical or conceptual context.

### 5. Use examples as proofs of understanding

A short example is often the fastest way to turn an abstract API into something
usable. Keep examples small, real, dependency-light, and aligned with the actual
public API.

### 6. Make the prose enjoyable to read

Educational does not mean theatrical. It means the reader feels guided. Use
clean section names, specific nouns, active verbs, and forward momentum.

### 7. Combine prose and diagrams intentionally

Do not drop a Mermaid block into a doc without interpretation. Introduce the
diagram, tell the reader what question it answers, and connect it back to the
surrounding explanation.

## Source Mapping Rules

Generated README files are outputs, not authoring targets.

When the target documentation is generated:

- treat the README as a compiled artifact,
- identify which source files own each section of the public story,
- edit those sources,
- regenerate docs,
- compare the output against the intended teaching experience.

Use the detailed checklist in
[Generated README source mapping checklist](./assets/source-mapping-checklist.md).

## External References And Media

When external references improve the docs, they must also improve trust.

Mermaid Markdown should be the default visual tool because it is text-based,
diff-friendly, and easier to keep synchronized with the code than static images.

When a document is expected to be read both on GitHub and in the generated docs
site, optimize Mermaid for GitHub compatibility first and treat local HTML
enhancements as secondary.

### For text sources

- Cite the source in a compact, readable way.
- Name the author or responsible organization.
- Link the canonical URL.
- If you quote or closely paraphrase, make the attribution explicit.
- Prefer synthesis over copying.

### For Wikipedia articles

- Prefer citing them as background reading, not as the primary authority for
  repo-specific behavior.
- Attribute to `Wikipedia contributors` unless a named author is clearly more
  appropriate.
- Treat Wikipedia as a learning bridge, not a substitute for source code.

### For Wikimedia Commons media

- Prefer Commons-hosted media over images embedded only on a Wikipedia article.
- Verify the file description page directly.
- Credit the original creator, not merely the uploader.
- Record the exact license and link to it when required.
- Respect ShareAlike obligations for derivative works.
- Check for non-copyright restrictions such as personality, trademark, or
  moral-rights concerns.
- If anything is unclear, do not use the image.

Use the operational checklist in
[External sources and Wikimedia media guide](./assets/external-sources-and-media.md).

## Mermaid Markdown

Prefer Mermaid for diagrams that can live directly in Markdown or generated
README surfaces.

When styling is available, default to the palette and contrast rules in
[Astro Bird visual style guide](./assets/visual-style-guide.md).

Good defaults:

- `flowchart` for architecture, decision flow, and orchestration,
- `sequenceDiagram` for protocol and lifecycle ordering,
- `stateDiagram-v2` for stateful runtime behavior,
- `classDiagram` for public type relationships,
- `erDiagram` for domain and storage relationships,
- `xychart-beta` for simple metrics and trend views,
- Markdown tables when a table is clearer than a diagram.

Fast recipe index:

- boundaries and subsystem maps: `flowchart`, with `architecture-beta` or
  `block` when grouping or lane control matters more,
- ordered handoffs and protocol timing: `sequenceDiagram`,
- ownership and API structure: `classDiagram`,
- lifecycle and runtime modes: `stateDiagram-v2`,
- multiplicity and entity relationships: `erDiagram`,
- reader or operator path: `journey`,
- history or milestones: `timeline`,
- active work plan and dependencies: `gantt`,
- two-axis prioritization: `quadrantChart`,
- proportional or trend views: `pie`, `xychart-beta`, or `sankey` depending on
  whether the story is share, trend, or weighted flow,
- branching, releases, and migrations: `gitGraph`,
- concept clustering or teaching taxonomy: `mindmap`.

Use Mermaid because it helps documentation resist diagram rot: the diagrams stay
textual, reviewable, and close to the code they explain.

For diagram selection, syntax caveats, and validation guidance, use
[Mermaid diagram playbook](./assets/mermaid-diagram-playbook.md).

## Guardrails

- Do not hand-edit generated `src/**/README.md` files.
- Do not add external references that drown out the repo's own explanations.
- Do not add decorative images with no teaching value.
- Do not add decorative Mermaid blocks with no teaching value.
- Do not assume Mermaid features that work in local generated HTML will render
  the same way on GitHub.
- Do not choose cutting-edge Mermaid syntax when a more conservative diagram
  would teach the same idea.
- Do not assume a Wikimedia uploader is the author.
- Do not reuse non-free, fair-use, `-NC`, or `-ND` media.
- Do not force Mermaid to express a table or heatmap when Markdown or prose is
  clearer.
- Do not paste long quote blocks when a concise summary would teach better.
- Do not let a citation style overwhelm readability.

## Expected Final Output

A strong educational-docs pass should report:

- the target surface,
- the source files that were actually edited,
- the reader questions the pass was designed to answer,
- any Mermaid diagrams or tables added and why they were chosen,
- whether docs were regenerated,
- any external sources or media added, with attribution and license notes,
- any remaining doc gaps or follow-up opportunities.

## Companion Agent Contract

If a companion agent uses this skill, it should:

1. Name this skill explicitly as `educational-docs`.
2. Pass concrete reconnaissance findings into the skill instead of paraphrasing
  them away.
3. Keep the agent prompt focused on read-only discovery and gap reporting when
  the agent is a scout.
4. Avoid restating the full workflow, tone model, or guardrails that already
  live here.
5. Update the agent when this skill changes materially so both remain aligned.
