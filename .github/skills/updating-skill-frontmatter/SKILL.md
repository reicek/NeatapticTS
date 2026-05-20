---
name: updating-skill-frontmatter
description: 'Use when: updating SKILL.md YAML frontmatter, names, descriptions, argument hints, user-invocable flags, compatibility notes, or resource links.'
argument-hint: 'Name the skill folder, frontmatter fields to update, desired trigger scope, visibility decision, and validation command.'
user-invocable: false
disable-model-invocation: false
---

# Updating Skill Frontmatter

Use this skill to make small, safe skill metadata edits.

Rules:
- Preserve the skill's existing intent unless the user asked for a split or deprecation.
- Keep descriptions concise, discoverable, and scoped to the skill's actual job.
- Add or refine `argument-hint` when a skill is broad enough to need task shaping.
- Keep skill names stable unless the old name is clearly wrong and references can be updated.
- Validate with `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict`.

Return skill files changed, trigger-scope decision, validation evidence, and follow-up eval needs.