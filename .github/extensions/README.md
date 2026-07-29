# NeatapticTS Extension Catalog Pattern

This document describes how future NeatapticTS contributors can package reusable workflow pieces by borrowing concepts from the public Spec-Kit extension/preset model and mapping them onto the existing agent/skill/MCP architecture.

## Spec-Kit layering model

Spec-Kit composes workflow behavior in layers:

1. **Core templates** — `specify` command prompts and artifact templates live in `.specify/templates/`.
2. **Extensions** — an `extension.yml` manifest plus `commands/*.md` files (and optional `scripts/` and `config-template.yml`) registers new slash commands and `before_*`/`after_*` hooks. For example, the `git` extension adds branch-management commands that auto-fire around every core command.
3. **Presets** — a `preset.yml` manifest plus command/template overrides layers organizational policy on top of core templates. Lower priority numbers win. Presets are discovered through catalogs and can be stacked.
4. **Bundles** — distribution groupings of one or more extensions and presets. Bundles are the coarsest grain of the layering model (for example, a "team-workflow" bundle that ships both git hooks and a compliance preset).
5. **Project-local overrides** — `.specify/templates/overrides/` wins over all installed layers for a given file name.

Resolution is per file name: the first layer that provides `plan-template.md` wins. Composition strategies include `replace`, `prepend`, `append`, and `wrap` around `{CORE_TEMPLATE}`.

## Mapping to NeatapticTS concepts

| Spec-Kit concept                                                | NeatapticTS equivalent                                                                                                                        | Where it lives                                                           |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `extension.yml` manifest (id, version, tags, requires/provides) | `.agent.md` frontmatter + the generated routing table                                                                                         | `.github/agents/*.agent.md`, `.github/agent-skill-routing-table.md`      |
| Command files (`commands/*.md`)                                 | Numbered phase agents and specialist agents invoked by goal                                                                                   | `.github/agents/*.agent.md` bodies                                       |
| Templates (`templates/*.md`)                                    | Skill markdown files loaded into agent context                                                                                                | `.github/skills/*/SKILL.md`                                              |
| Presets (priority-sorted overrides)                             | Project-level custom instructions and shared skills; lower-level skills can be shadowed by higher-priority local instructions                 | `.github/copilot-instructions.md`, `.github/skills/*`, plan step packets |
| Bundles (multi-extension distributions)                         | A folder under `.github/extensions/` that groups related agent/skill sets; bundles are named groupings of extensions and presets              | `.github/extensions/<bundle-name>/`                                      |
| Catalogs                                                        | The generated routing table and MCP dispatch inventory                                                                                        | `.github/agent-skill-routing-table.md`, `neataptic-dispatch-mcp`         |
| Hooks (`before_specify`, `after_implement`, …)                  | Explicit phase orchestration; there is no generic hook system, but a future `.github/agent-hooks.yml` could provide the same extension points | `execute` skill, step-packet YAML                                        |

### How to read the mapping

- An **agent** is the closest analog to a Spec-Kit **extension**: the frontmatter declares metadata (`name`, `tier`, `skills`, `agents`, `tools`) and the markdown body is the command prompt.
- A **skill** is the closest analog to a Spec-Kit **preset/template**: it supplies reusable context (conventions, decision trees, guardrails) that agents load by name.
- The **routing table** is the repo-owned catalog: it lists every agent, its tier, its allowed sub-agents, and its attached skills. `neataptic-dispatch-mcp` answers "can this caller dispatch that target?" the same way Spec-Kit answers "is this extension compatible and installed?".
- The **MCP dispatch tier graph** is stronger than an extension manifest: it enforces that delegation only flows downward by tier, that only Tier-1 agents are user-invocable, and that every dispatch target is validated before a sub-agent is spawned.

## Adding a new contribution

If you want to add a reusable workflow piece:

1. **For a new phase-like command**, create a new `.github/agents/<name>.agent.md` file (or propose a Tier-1 phase agent) and regenerate the routing table with `npm run agents:routing-table`.
2. **For shared conventions or guardrails**, create a new `.github/skills/<skill-name>/SKILL.md` and attach it to the relevant agents in their frontmatter.
3. **For a project-local policy override**, edit `.github/copilot-instructions.md` or the active plan step packet rather than forking an agent file.
4. **For a curated multi-agent/skill bundle**, add a sub-folder under `.github/extensions/<bundle-name>/README.md` describing the bundle, and link to the real agent/skill files that make it up.

## Example `extension.yml` manifest

If the existing `neataptic-dispatch-mcp` tooling ever consumes Spec-Kit-style manifests, a hypothetical NeatapticTS-compatible extension could look like this:

```yaml
schema_version: '1.0'

extension:
  id: 'neataptic-bug-triage'
  name: 'NeatapticTS Bug Triage'
  version: '1.0.0'
  description: 'Assess → fix → test workflow for bug reports, aligned with NeatapticTS phase agents.'
  author: 'neataptic'
  repository: 'https://github.com/reicek/NeatapticTS'
  license: 'MIT'

requires:
  neataptic_version: '>=2.0.0'
  tools:
    - name: git
      required: false

provides:
  commands:
    - name: 'bug.assess'
      agent: 'failure-triage-specialist'
      description: 'Assess a bug report and write evidence without editing source.'
    - name: 'bug.fix'
      agent: 'test-fix-workflow'
      description: 'Produce a targeted fix for an assessed bug.'
    - name: 'bug.test'
      agent: 'unit-test-writer'
      description: 'Add regression tests that should fail before the fix and pass after.'

  skills:
    - name: 'triaging-test-failures'
    - name: 'test-fix-workflow'

  tags:
    - 'bug'
    - 'triage'
    - 'neataptic'
```

## Compatibility note

Spec-Kit extensions are designed to be discovered, installed, and layered at runtime through external catalogs. NeatapticTS keeps the same **catalog** idea but makes it **repo-owned and deterministic**: the generated `.github/agent-skill-routing-table.md` plus the `neataptic-dispatch-mcp` server act as the compatibility layer.

A Spec-Kit `extension.yml` declares what commands and hooks an extension provides. The NeatapicTS MCP dispatch graph adds tier enforcement on top of that: a caller can only dispatch to a strictly higher tier, only Tier-1 agents may be user-invocable, and `build_dispatch_packet` must return `dispatch_allowed: true` before any sub-agent is spawned. In other words, the extension manifest says "what can run," while the tier graph and MCP dispatch say "who is allowed to invoke it."

When adding a new agent or skill, prefer updating the routing table and using `neataptic-dispatch-mcp` validation over hand-editing generated files or introducing ad-hoc command registrations.
