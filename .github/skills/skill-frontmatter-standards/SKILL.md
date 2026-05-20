---
name: skill-frontmatter-standards
description: 'Use when: validating or designing SKILL.md frontmatter, skill folder names, argument-hint fields, descriptions, visibility, compatibility text, or local resource links.'
argument-hint: 'Describe the skill folder, intended trigger scope, visibility decision, argument hint, local resources, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Skill Frontmatter Standards

Use this skill when a task changes `.github/skills/<skill-name>/SKILL.md` metadata.

Rules:
- Skill folder names and `name` values must match exactly and use lowercase kebab-case.
- Descriptions are the discovery surface; start with clear trigger language such as `Use when:` when possible.
- Keep descriptions under the Agent Skills 1024-character limit.
- Provide `argument-hint` for every repo skill so prompts stay compact and task-shaped.
- Set `user-invocable` intentionally; hide narrow helper skills with `false`.
- Keep scripts, templates, and resources inside the owning skill folder when they are skill-specific.
- Run `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict` after metadata edits.

Return files changed, metadata decisions, validation command, and residual routing risk.