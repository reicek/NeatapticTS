---
name: skill-output-evals
description: 'Evaluate NeatapticTS skill output quality with evidence-backed assertions. Use when comparing with-skill versus baseline behavior, grading specialist output, aggregating pass rates, or deciding whether a skill improves quality enough to keep.'
argument-hint: 'Describe the skill, eval fixtures, expected outputs, assertions, and baseline or previous version.'
user-invocable: false
disable-model-invocation: false
---

# Skill Output Evals

Use this skill when a skill needs quality evidence beyond manual inspection.

## Workflow

1. Define each eval with a realistic prompt, expected output, optional input files, and objective assertions.
2. Compare the current skill against no-skill or a previous skill snapshot.
3. Grade assertions with concrete evidence, not vibes.
4. Use scripts for mechanical checks such as file existence, JSON validity, or countable output.
5. Aggregate pass rate, timing, token cost when available, and failure categories.
6. Feed failures back into skill instructions without overfitting to one prompt.

## Sources

- Agent Skills evaluation guidance recommends with-skill/baseline comparisons, assertion evidence, aggregation, and human review for qualitative gaps.