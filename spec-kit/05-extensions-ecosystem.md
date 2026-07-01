# Spec Kit — Extensions, Presets & Ecosystem

Spec Kit's core is intentionally small. Most workflow behavior (git, bug triage, agent-context injection, self-test) lives in extensions. Organizational variation is layered through presets. This section describes how the extension/preset system works and how it compares with NeatapticTS's skill/agent architecture.

## Extension Model

Each extension is a folder with:

- `extension.yml` — metadata, dependencies, commands, hooks, tags
- `commands/*.md` — prompt files registered as slash commands
- `scripts/` (optional) — platform-specific Bash / PowerShell helpers
- `config-template.yml` (optional) — per-extension configuration

Extensions declare compatibility:

```yaml
requires:
  speckit_version: ">=0.9.0"
```

Extensions provide commands:

```yaml
provides:
  commands:
    - name: speckit.bug.assess
      file: commands/speckit.bug.assess.md
      description: "Assess a bug report..."
```

Extensions also declare hooks that fire before/after core commands. The git extension demonstrates the full hook map:

- `before_constitution`
- `before_specify`, `before_clarify`, `before_plan`, `before_tasks`, `before_implement`, `before_checklist`, `before_analyze`, `before_taskstoissues`
- `after_constitution`, `after_specify`, `after_clarify`, `after_plan`, `after_tasks`, `after_implement`, `after_checklist`, `after_analyze`, `after_taskstoissues`

This is a formal, event-driven plugin system. Hooks can be optional (prompt the user) or mandatory.

---

## Bug Extension (`extensions/bug/`)

The bug extension is a full triage workflow embedded in three commands:

- `/speckit.bug.assess`
- `/speckit.bug.fix`
- `/speckit.bug.test`

### `/speckit.bug.assess`

This command is the most detailed prompt in the entire bug extension. It ingests pasted text or a URL, applies a **URL Trust Policy**, writes a structured `assessment.md` to `.specify/bugs/<slug>/`, and reports:

- Verdict: `valid` | `likely valid, needs reproduction` | `invalid`
- Severity: `critical` | `high` | `medium` | `low`

It never modifies source files during assessment. The URL Trust Policy is worth emulating:

- **Refuse outright**: non-`http(s)`, loopback, RFC1918 private space, cloud metadata endpoints
- **Fetch without prompting**: `github.com`, `gitlab.com`, `bitbucket.org`, Jira, Linear, Stack Overflow, Sentry
- **Otherwise**: interactive users must confirm; automated mode skips and records `[UNVERIFIED]`

### Verdict and Severity

Severity is assigned with rationale covering:

- user impact
- blast radius
- data risk
- regression vs. long-standing

This is similar in spirit to NeatapticTS's triaging-test-failures skill but more structured.

---

## Git Extension (`extensions/git/`)

The git extension manages the repository boundary around SDD artifacts:

- `/speckit.git.feature` — create `###-short-name` or timestamped feature branch
- `/speckit.git.initialize` — init repo + initial commit
- `/speckit.git.validate` — branch naming validation
- `/speckit.git.remote` — detect remote URL for GitHub integration
- `/speckit.git.commit` — auto-commit after commands

### Hook Integration

`before_specify` creates the feature branch. `before_clarify/plan/tasks/implement/checklist/analyze` optionally commit outstanding changes. `after_*` optionally commit command outputs. This keeps the working tree clean and gives every SDD artifact a git trail.

### Configuration Priority

Branch numbering is resolved in order:

1. `.specify/extensions/git/git-config.yml`
2. `.specify/init-options.json` `feature_numbering`
3. `.specify/init-options.json` `branch_numbering` (deprecated)
4. Default `sequential`

This layered config pattern is repeated for presets and agent-context.

---

## Agent-Context Extension (`extensions/agent-context/`)

The agent-context extension injects the current plan path into the coding agent's instructions file (e.g., `.github/copilot-instructions.md`, `CLAUDE.md`, `AGENTS.md`).

### Managed Section

It reads `agent-context-config.yml` to find `context_file` / `context_files` and wraps its contribution in configurable markers:

> "Defaults to `<!-- SPECKIT START -->` and `<!-- SPECKIT END -->` when the field is missing."

The extension then auto-detects the most recently modified `specs/*/plan.md` and writes a managed block so the coding agent always has a pointer to the active plan.

### Trust Boundary

The prompt explicitly rejects dangerous paths:

> "Context file paths must stay project-relative; absolute paths, Windows drive paths, backslash separators, and `..` path segments are rejected."

This is a good example of treating the agent context file as a shared, managed surface rather than an ad-hoc append target.

---

## Self-Test Extension (`extensions/selftest/`)

`/speckit.selftest.extension <name>` simulates the full extension lifecycle:

1. Catalog discovery (`specify extension info <name>`)
2. Installation (`specify extension add <name>`)
3. Registration verification (`cat .specify/extensions/.registry/<name>.json`)
4. Terminal-style pass/fail report

The command treats the `install_allowed: false` catalog flag correctly: direct add may be expected to fail, in which case it falls back to the catalog `download_url`.

This is a developer-experience command, not a production workflow command.

---

## Presets

Presets override templates and commands without changing tooling. They are layered with priority numbers: lower wins.

### Resolution Stack

From highest to lowest precedence:

1. Project-local overrides — `.specify/templates/overrides/`
2. Installed presets — `.specify/presets/<id>/`
3. Installed extensions — `.specify/extensions/<id>/`
4. Spec Kit core — `.specify/templates/`

Each **file name** is resolved independently, so different files can come from different layers. The documentation explains composition strategies:

- **replace** — first match wins
- **prepend** — preset content before lower-priority content
- **append** — preset content after lower-priority content
- **wrap** — replace `{CORE_TEMPLATE}` with lower-priority content
- Scripts support **replace** and **wrap** via `$CORE_SCRIPT`

### Preset CLI

```text
specify preset search [query]
specify preset add <id> --priority <N> --from <url> --dev <path>
specify preset remove <id>
specify preset list
specify preset info <id>
specify preset resolve <name>
specify preset enable|disable <id>
specify preset set-priority <id> <N>
```

### Catalogs

Presets are discovered through catalogs with priority and install permissions:

- Environment variable `SPECKIT_PRESET_CATALOG_URL` overrides everything
- `.specify/preset-catalogs.yml` (project)
- `~/.specify/preset-catalogs.yml` (user)
- Built-in official + community catalogs

Catalog entries can be `install_allowed: false` for discovery-only catalogs.

---

## Lean Preset

The `lean` preset ships minimal versions of the core commands:

- `speckit.specify.md`
- `speckit.plan.md`
- `speckit.tasks.md`
- `speckit.implement.md`
- `speckit.constitution.md`

For example, the lean `speckit.plan.md` outline is:

```text
1. Read `.specify/feature.json`
2. Load context: constitution and spec
3. Create an implementation plan in plan.md
   - Technical context, design decisions, architecture, file structure
```

This shows how presets can compress the full prompt down to an outline while keeping the same command name.

---

## Comparison with NeatapticTS Skill/Agent Architecture

| Aspect | Spec Kit | NeatapticTS |
|--------|----------|-------------|
| **Modular units** | Extensions (`extension.yml` + commands + scripts) | `.agent.md` agents + `SKILL.md` skills |
| **Command registration** | Extensions/presets register slash commands automatically | Agent frontmatter declares skills; routing table routes by goal/phase |
| **Workflow hooks** | Formal `before_*` / `after_*` hooks in `extension.yml` | Phase orchestrators dispatch explicitly; no generic hook system |
| **Layered overrides** | Presets + project-local overrides, priority-sorted | `.github/copilot-instructions.md` + custom instructions in `.agent.md`; less granular file-level composition |
| **Config per plugin** | `git-config.yml`, `agent-context-config.yml` | No per-skill YAML config convention; config embedded in skill markdown |
| **Catalog / discovery** | Official + community catalogs for presets and extensions | No external catalog; all customizations live in repo |
| **Self-test** | `/speckit.selftest.extension` validates lifecycle | No equivalent skill or agent |
| **Agent context injection** | `agent-context` extension writes managed block into copilot instructions | `.agent.md` frontmatter loaded by Copilot; no runtime injection of active plan path |
| **Branch naming / git** | Git extension automates branch creation and commit hooks | No standard git workflow skill; agent actions use git ad-hoc |

### What NeatapticTS Does Better

- **Tiered orchestration**: seven numbered phase agents provide a clear SDLC contract with gates.
- **Machine-readable plan packets**: YAML step blocks with validation commands are more actionable than Markdown checkboxes.
- **Learning events**: ISO 42001-style structured events are captured explicitly (see `06-iso42001-learning-overlap.md`).
- **No external catalogs**: all behavior is repo-owned, reproducible, and versioned.

### What Spec Kit Does Better

- **Clean separation of workflow hooks**: extensions can insert behavior before/after commands without rewriting the core prompt.
- **Preset composition**: organizations can layer policy presets (compliance, team-workflow) without forking the repo.
- **Managed agent context**: auto-updating a delimited block in `copilot-instructions.md` keeps the agent aligned with the active plan.
- **Self-test command**: a first-class DX feature for validating extension/preset behavior.

## Recommended Assimilation

1. **Add a lightweight hook/event model** to NeatapticTS: define pre/post phase hooks in `.github/agent-hooks.yml` so extensions can run `git commit` or context updates without embedding git logic in every agent.
2. **Create a `copilot-instructions.md` managed block** for active plan injection, similar to the agent-context extension.
3. **Keep NeatapticTS's repo-owned customization model**; do not introduce external catalogs unless there is a governance need.
4. **Consider a self-test skill** that validates agent/skill lifecycle the way `/speckit.selftest.extension` validates extensions.
5. **Borrow the URL Trust Policy** from `/speckit.bug.assess` for any agent that fetches external bug reports or references.
