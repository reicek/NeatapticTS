---
name: skill-description-evals
description: 'Design and grade trigger evals for NeatapticTS Agent Skill and custom agent descriptions. Use when testing should-trigger and should-not-trigger queries, preventing broad false positives, or optimizing descriptions under the 1024-character limit.'
argument-hint: 'Name the skill or agent, describe expected trigger scope, and provide observed trigger results or planned eval queries.'
user-invocable: false
disable-model-invocation: false
---

# Skill Description Evals

Use this skill when a description controls whether an agent or skill activates.

## Workflow

1. Create realistic should-trigger and should-not-trigger queries.
2. Include casual phrasing, typos, file paths, near misses, and multi-step prompts.
3. Split query sets into train and validation groups when optimizing descriptions.
4. Avoid adding exact failed-query keywords as one-off fixes.
5. Keep descriptions concise and below 1024 characters for skills.
6. Record trigger rates and false-positive categories in the active plan.

## Sources

- Agent Skills optimizing descriptions guidance recommends realistic trigger queries, negative near misses, repeated runs, and train/validation splits.