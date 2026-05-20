---
name: green-validation-gates
description: 'Run and interpret green-phase validation for NeatapticTS agent workflow changes. Use when deciding focused test order, rerouting failed checks, enforcing coverage guard, or proving scripts, plans, agents, and skills are consistent.'
argument-hint: 'Describe changed files, expected validations, latest failures, and whether strict customization validation should pass yet.'
user-invocable: false
disable-model-invocation: false
---

# Green Validation Gates

Use this skill after implementation edits.

## Workflow

1. Run the narrowest validation that matches the changed surface.
2. For customization scripts, run each script in audit mode and targeted strict mode when the target state should hold.
3. For plan-only changes, run markdown whitespace checks and plan-sync validation.
4. For `.github/agents/`, run agent frontmatter and graph validation.
5. For `.github/skills/`, run skill frontmatter validation.
6. Route failures back to Red Testing or Implementation instead of continuing to Documentation.
7. Record validation evidence in the active plan.

## Repo Gates

- Run `npm run build` or `npm run lint` when TypeScript, package scripts, or source code changes require it.
- Run `npm run docs` only when docs-generation inputs are touched.