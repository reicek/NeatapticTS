# 04 — Templates and Artifacts

## External Templates

List any templates, boilerplates, or scaffold generators the repo ships.

## File Layout Conventions

| Surface      | External Layout      | NeatapticTS Layout                               |
| ------------ | -------------------- | ------------------------------------------------ |
| Agent spec   | <file path / format> | `.github/agents/*.agent.md` with frontmatter     |
| Skill spec   | <file path / format> | `.github/skills/*/SKILL.md` with frontmatter     |
| Plan tracker | <file path / format> | `plans/*.plans.md` with `PlanUpdate` YAML blocks |
| Templates    | <file path / format> | `.github/templates/*/`                           |
| Gate script  | <file path / format> | `scripts/agent-customization/gates/*.gate.mjs`   |

## Traceability IDs

Does the external tool use unique IDs, slugs, or tags to link artifacts? How
does that compare with NeatapticTS `slice_id` and plan-update references?

## Cherry-Pick Candidates

- <Template or layout worth adopting and why>
- <Template or layout to reject and why>
