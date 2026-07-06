# agent-json-body-to-md skill load fix

**Status:** [DONE]

## Scope

Fix the YAML frontmatter of `.github/skills/agent-json-body-to-md/SKILL.md` so it loads correctly in strict YAML parsers. The description contained an unescaped apostrophe inside a single-quoted YAML scalar, which breaks standard YAML loaders.

## Root cause

Line 3 of the original frontmatter:

```yaml
description: 'Use when: converting an agent file's JSON body to Markdown prose.'
```

The apostrophe in `file's` terminates the single-quoted YAML scalar prematurely. A strict YAML parser throws an error and the skill fails to load. The repo's custom frontmatter parser tolerated it, so `validate-skill-frontmatter` did not surface the issue.

## Fix

Reworded the description to remove the apostrophe entirely, keeping the skill's trigger scope intact:

```yaml
description: 'Use when: converting the JSON body of an agent file to Markdown prose.'
```

## Files changed

- `.github/skills/agent-json-body-to-md/SKILL.md` — fixed frontmatter description.
- `.github/agent-skill-routing-table.md` — regenerated because the source hash changed.
- `plans/README.md` — added the new tracker plan to the active plans list.
- `plans/Roadmap.md` — added the new tracker plan under the Holistic Agent & Skill Optimization meta-workflow lane.

## Final state

- The skill frontmatter description was reworded to remove the apostrophe, so strict YAML parsers load the file correctly.
- The canonical routing table was regenerated and the skill is included with the expected source hash.
- All repo validators pass and the tracker plan is closed.

## Audit summary

- `.github/skills/agent-json-body-to-md/SKILL.md` — fixed frontmatter description.
- `.github/agent-skill-routing-table.md` — regenerated because the source hash changed.
- `plans/README.md` and `plans/Roadmap.md` — index entries updated and archived.
- `plans/completed/agent-json-body-to-md-skill-load-fix.logs.md` — compressed done-state record created.

## Reopen conditions

Reopen if the `agent-json-body-to-md` skill frontmatter drifts and strict YAML parsing fails again.

## Audit log

See the same-boundary `agent-json-body-to-md-skill-load-fix.logs.md` for full validation history and the durable done-state record.
