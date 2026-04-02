# README Tone Model

Use this rubric when shaping JSDoc so generated README output feels closer to
`examples/flappy_bird/README.md` and farther from a dry symbol dump.

## What Makes The Flappy Bird README Strong

### It opens with purpose

The first paragraphs tell the reader what the folder is for and why it matters.
That creates immediate orientation.

### It teaches through practical questions

The document frames the system around questions a curious engineer would ask:

- how training connects to a repeatable control problem,
- how evaluation reduces luck,
- how browser playback works without moving simulation to the main thread,
- how the evolved network becomes inspectable.

That style makes the reader feel invited into the design instead of lectured at.

### It gives a map before detail

The README uses sections such as folder map, layers, execution paths, design
choices, and recommended reading order. That sequence helps the reader build a
mental model before they dive into files.

### It explains tradeoffs openly

The prose does not pretend the architecture is arbitrary. It explains why the
trainer uses staged evaluation, why the browser avoids simulation authority, and
why compatibility facades still exist.

### It respects different reading goals

A new reader gets orientation. A focused reader gets entry points for training,
rendering, or environment logic. This is a strong model for public-facing docs.

## Rubric

A generated README is close to the target bar when it does most of the
following.

1. Opens with a clear statement of role and value.
2. States what the module is for before listing files or exports.
3. Uses section names that help a reader navigate, not just catalog.
4. Explains architecture, flow, or responsibility boundaries when the topic is
   non-trivial.
5. Mentions important defaults, invariants, and design choices.
6. Includes examples or usage cues where a new reader would otherwise guess.
7. Recommends where to read next when the surface is broad.
8. Feels specific to the module rather than interchangeable with any library.

## Common Failure Modes

- Opening with file listings before purpose.
- Repeating type signatures in prose.
- Explaining what exists without explaining why it exists.
- Omitting the reader's likely next question.
- Treating README generation as an excuse for generic comments.
- Adding "educational" fluff that does not improve understanding.

## Practical Rewrite Moves

When a section feels weak, use one of these moves:

- Replace "This file contains..." with "This module is responsible for..."
- Add one sentence that explains why the boundary exists.
- Convert a passive list of exports into a reading path or execution path.
- Add a short "What this is for" paragraph before technical detail.
- Add a "Design choices" section when the architecture has visible tradeoffs.
- Add a "Recommended reading order" section when the surface is broad.
