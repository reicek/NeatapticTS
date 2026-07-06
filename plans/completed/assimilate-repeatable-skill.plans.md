# External Tool Assimilation — Repeatable Skill/Agent

**Status:** [DONE]

```yaml
phase: 1
title: 'External Tool Assimilation — Repeatable Skill/Agent'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/assimilate-repeatable-skill.plans.md'
copy_paste: true
next_phase: 'null — workstream complete; see plans/completed/assimilate-repeatable-skill.logs.md'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'planning-acceptance-criteria'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md'
acceptance_criteria:
  - 'Step packets for the phase are authored and pass step-packet gate'
  - 'README and Roadmap contain active entries for this workstream'
placeholder_steps:
  - 'Step 01 — Planning the skill/agent boundary and process'
  - 'Step 02 — Research existing repo-fetch patterns and validate tooling'
  - 'Step 03 — Write red tests / fixtures for the assimilation script'
  - 'Step 04 — Implement skill, agent, templates, and script'
  - 'Step 05 — Green validation / sample assimilation run'
  - 'Step 06 — Documentation and usage examples'
  - 'Step 07 — Session logging and tracker closure'
```

## Scope

Make the ad-hoc Spec Kit analysis (`spec-kit/`) a repeatable, repo-owned capability.
The goal is a slash-command-style flow (`/assimilate https://github.com/someAwesomePlugin`)
that any Tier 1 orchestrator can invoke to study an external GitHub repository,
persist its important surfaces, compare them with NeatapticTS, and produce actionable
cherry-pick recommendations.

### What the flow produces

For a given public GitHub repository URL:

1. A folder under the project root named after the repo (user-supplied or auto-derived).
2. A `verbatim/` subfolder containing full copies of important source files.
3. Per-area summary files (`00-overview.md`, `01-workflow.md`, `02-validation.md`, ...).
4. A final condensed `<folder>.md` synthesis with top recommendations.
5. A license/attribution record that honors the external repo's license.

### What stays the same

- The NeatapticTS phase-orchestrated SDLC (`01-planning` → `07-logging`), gate contracts,
  step-packet YAML, and ISO 42001 learning-event logging remain unchanged.
- No `src/` library code is modified.
- No changes to MCP server implementations.

### Naming and tier placement

| Element                  | Proposed name                | Tier  | Why                                                                                                    |
| ------------------------ | ---------------------------- | ----- | ------------------------------------------------------------------------------------------------------ |
| Skill (durable workflow) | `external-tool-assimilation` | skill | Procedure + templates + comparison heuristics                                                          |
| Specialist agent         | `assimilator`                | 3     | Narrow, reusable job: analyze external repo and return a bounded study                                 |
| User-facing trigger      | `/assimilate <repo-url>`     | —     | Phrase in skill/agent descriptions; actual invocation routes through `02-researching` or `01-planning` |

The skill is **not** `user-invocable`. Orchestrators (`01-planning`, `02-researching`) detect
the trigger phrase and delegate to the `assimilator` specialist. The `assimilator` returns
a structured study; the orchestrator records it in `plans/` or `docs/research/` as appropriate.

### Alignment with active lanes

- This workstream creates a new skill and a new Tier 3 specialist, so it conceptually
  overlaps the surfaces covered by the **Holistic Agent & Skill Optimization** lane.
  That lane is now **[DONE]** and archived at
  `plans/completed/holistic-agent-skill-optimization.plans.md`, so there is no active
  owner to race against.
- Step 02 research proceeded independently because it did not modify `.github/agents/`
  or `.github/skills/`.
- Step 04 implementation must follow the post-Holistic agent/skill frontmatter and
  delegation standards when creating new `.agent.md` / `SKILL.md` files and
  regenerating `.github/agent-skill-routing-table.md`.

---

## Current state

Claim: 04-implementing — slice-fix on `scripts/assimilation/assimilate-repo.mjs` synthesis filename.

- We have a complete manual example under `C:\NeatapticTS\spec-kit\`:
- `spec-kit.md` — final synthesis with 10 cherry-pick recommendations.
- `00-overview.md` through `07-verbatim-phrases-and-verbs.md` — per-area analyses.
- `spec-kit\verbatim\` — 31 full source files from the `github/spec-kit` repo.
- `archive\` — prior exploratory drafts.
- The manual process used a mix of raw GitHub fetches, blob API calls, tree API calls,
  and local comparison against NeatapticTS agents/skills/plans.
- No reusable automation, template, or agent definition exists yet.

---

## Implementation phases

### Phase 1 — External Tool Assimilation [DONE]

```yaml
phase: 1
title: 'External Tool Assimilation'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/assimilate-repeatable-skill.plans.md'
copy_paste: true
next_phase: 'null — workstream complete; see plans/completed/assimilate-repeatable-skill.logs.md'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'agent-frontmatter-standards'
  - 'skill-frontmatter-standards'
  - 'model-routing-and-budget'
  - 'license-attribution-audit'
  - 'planning-acceptance-criteria'
  - 'plan-sync-validation'
  - 'research-methodology'
  - 'execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md'
acceptance_criteria:
  - 'Plan file is registered in plans/README.md and plans/Roadmap.md with a consistent [WIP] status'
  - 'Phase/step YAML blocks pass plan-sync and step-packet gates'
  - 'Skill name, agent name, tier, and trigger phrase are unambiguous'
placeholder_steps:
  - 'Step 01 — Planning the skill/agent boundary and process'
  - 'Step 02 — Research existing repo-fetch patterns and validate tooling'
  - 'Step 03 — Write red tests / fixtures for the assimilation script'
  - 'Step 04 — Implement the skill, agent, templates, and script'
  - 'Step 05 — Green validation and sample assimilation'
  - 'Step 06 — Documentation and usage examples'
  - 'Step 07 — Session logging and tracker closure'
```

[DONE] Phase 1: implemented the `external-tool-assimilation` skill, the `assimilator` Tier-3 specialist, eight comparison templates, the `scripts/assimilation/assimilate-repo.mjs` script, red-to-green tests (10/10 pass), and a sample end-to-end assimilation of `github/choosealicense.com`. All plan-sync, step-packet, agent-graph, routing-table, frontmatter, and lint gates passed. Detailed evidence moved to `plans/completed/assimilate-repeatable-skill.logs.md`.

**Stop conditions:**

- **Done:** all seven step packets pass their gates, index/roadmap entries are consistent, and the sample assimilation run produces the expected artifacts.
- **Blocked:** a gate or sample run fails; route back to the smallest relevant step.

#### Step 01 — Planning the skill/agent boundary and process [DONE]

```yaml
phase: 1
step: 1
title: 'Planning the skill/agent boundary and process'
goal: 'planning'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 02 — Research existing repo-fetch patterns and validate tooling'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'agent-frontmatter-standards'
  - 'skill-frontmatter-standards'
  - 'model-routing-and-budget'
  - 'license-attribution-audit'
  - 'planning-acceptance-criteria'
  - 'plan-sync-validation'
  - 'research-methodology'
  - 'execute'
specialists:
  - 'plan-scout'
  - 'model-name-auditor'
validation:
  - 'node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
  - 'node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
acceptance_criteria:
  - 'Plan file exists at plans\assimilate-repeatable-skill.plans.md with Step 01-07 packets'
  - 'plans\README.md and plans\Roadmap.md contain active entries for this workstream'
  - 'plan-sync gate and step-packet gate pass'
  - 'Skill name, agent name, tier, and trigger phrase are unambiguous'
```

**User instruction:** Paste this full step packet.

**Step objective:** Define the exact boundary of the new skill and specialist agent,
restate the spec-kit process as a reusable machine-readable workflow, choose files and
templates, and produce the remaining Step 02-07 packets.

**Context the agent must know:**

- The canonical prior art lives under `C:\NeatapticTS\spec-kit\`.
- NeatapticTS agent/skill customization surfaces are governed by:
- `.github\agent-skill-routing-table.md` (generated)
- `scripts\agent-customization\validate-agent-frontmatter.mjs`
- `scripts\agent-customization\validate-skill-frontmatter.mjs`
- `scripts\agent-customization\validate-agent-graph.mjs`
- The **Holistic Agent & Skill Optimization** lane established the existing agent/skill
  inventory standards and is now [DONE]/archived; new skill/agent work should follow
  those standards and does not need to wait for an active lane owner.

**Execution steps:**

1. Read `spec-kit.md`, `00-overview.md`, `07-verbatim-phrases-and-verbs.md`, and the
   `spec-kit\verbatim\` layout.
2. Decide skill name, agent name, tier, model tier, and user-facing trigger phrase.
3. List every file to create or modify (skill, agent, templates, script, routing table).
4. Define the comparison dimensions and per-area summary file naming convention.
5. Define the validation approach and ISO 42001 learning-event notes.
6. Author Step 02-07 packets below this step.
7. Update `plans\README.md` and `plans\Roadmap.md` with active entries.
8. Run `validate-plan-phase-packets.mjs` and `validate-plan-sync.mjs`.

**Stop conditions:**

- **Done:** Step 02-07 packets exist, index/roadmap entries are consistent, gates pass.
- **Blocked:** The Holistic lane owner does not agree to the coordination model.
- **Route-back:** If a required file or assumption is missing, escalate to `00-helping`.

**Required validation:**

- `node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`

**Plan update requirement:** Update this file with the authored packets, decisions,
and evidence before ending.

#### Step 02 — Research existing repo-fetch patterns and validate tooling [DONE]

```yaml
phase: 1
step: 2
title: 'Research existing repo-fetch patterns and validate tooling'
goal: 'researching'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 03 — Write red tests / fixtures for the assimilation script'
skills:
  - 'research-methodology'
  - 'repo-cortex-workflow'
  - 'execute'
specialists:
  - 'repo-cortex-scout'
  - 'plan-scout'
validation:
  - 'Confirm GitHub raw, tree, and blob endpoints are reachable from the local environment'
  - 'Confirm node --version and npm scripts can run a small ESM fetch prototype'
  - 'Cortex search for existing fetch/util scripts in scripts/ and rag-index/'
acceptance_criteria:
  - 'A documented fetch strategy exists with raw, API, and git-clone fallbacks'
  - 'Rate-limit and 404 handling are specified'
  - 'No existing script already solves the same problem'
```

**User instruction:** Paste this full step packet.

**Step objective:** Validate that the repo-fetch pipeline used manually for spec-kit
(raw GitHub, tree API, blob API, git clone fallback) can be turned into a reliable
script, and confirm there are no reusable in-repo utilities that already do this.

**Context the agent must know:**

- The manual spec-kit pass used:
- `https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>` for individual files.
- `https://api.github.com/repos/<owner>/<repo>/git/trees/<ref>?recursive=1` for structure.
- `https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<ref>` as an API fallback.
- Some files were not fetchable via raw/API and were noted in `notes\fetch-failures.md`.
- Windows PowerShell is the local shell; use `node` ESM scripts, not shell pipelines.

**Execution steps:**

1. Search the repo for existing fetch/download utilities.
2. Prototype a minimal ESM script that fetches a GitHub repo tree and a few files.
3. Test rate-limit handling and 404 behavior.
4. Document the chosen fetch hierarchy and failure modes.
5. Update this plan with findings.

**Stop conditions:**

- **Done:** Fetch strategy is documented and prototyped.
- **Blocked:** GitHub API is unreachable or rate-limited in this environment.
- **Route-back:** If an existing utility already does this, route back to Step 01 to reuse it.

**Required validation:**

- Prototype fetch succeeds for at least one public repo.
- Prototype gracefully handles a missing file.

**Plan update requirement:** Add the fetch strategy to the plan and adjust Step 03/04
if a different implementation shape is needed.

**Step 02 research findings:**

- **No reusable in-repo utility exists.** A `repo-cortex-scout` survey of `scripts/`,
  `rag-index/`, `spec-kit/`, `docs/`, and `.github/` found no script that fetches an
  arbitrary public GitHub repo tree + files. Existing `fetch()` callers are limited to:
- `rag-index/download-model.mjs` and `rag-index/download-reranker.mjs`
  (HuggingFace single-asset downloads with SHA-256 verification and retry).
- `scripts/mcp-semantic/tools/turso-branch.mjs` and `turso-pitr.mjs`
  (authenticated Turso Platform API calls).
- `scripts/render-docs-html/render-docs-html.shared.ts` (builds GitHub links, does not fetch).
  There is no `scripts/assimilation/` folder and no `assimilate-repo*` file.

- **Chosen fetch hierarchy (priority order) for `scripts\assimilation\assimilate-repo.mjs`:**

1.  `https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>` — primary path for
    individual blobs. Highest rate-limit budget and no JSON decoding.
2.  `https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<ref>` — fallback
    when raw returns non-200 or non-text content. Decode `content` from base64.
3.  `https://api.github.com/repos/<owner>/<repo>/git/trees/<ref>?recursive=1` — used
    once per repo to enumerate the file tree and identify blobs to download.
4.  `git clone --depth 1 <repo-url> <folder>` — final fallback when the API rate-limit is
    exhausted, the repo contains LFS files, submodules, or a large number of raw/API
    failures. After cloning, copy/convert the working tree into the same verbatim/summary
    layout produced by the API path.

- **Rate-limit and 404 handling:**
- Read `x-ratelimit-remaining` and `x-ratelimit-reset` from every API response.
- If `remaining == 0`, sleep until `reset` (or fail fast if the user prefers `--no-wait`).
- Support optional `GITHUB_TOKEN` for 5,000 requests/hour; add `Authorization: token ${TOKEN}`
  to tree/contents API calls (raw does not need auth).
- On raw 404/403/5xx, immediately fall back to contents API; do not count a raw miss as a
  hard failure.
- On contents API 404, record the path in `<folder>\notes\fetch-failures.md` and continue.
- Add a small concurrency limit (default 5 parallel fetches) with exponential back-off to
  avoid tripping secondary GitHub throttles.

- **Prototype evidence:**
- File: `tmp\fetch-prototype.mjs`
- Command: `node tmp\fetch-prototype.mjs https://github.com/github/spec-kit C:\NeatapticTS\tmp\fetch-prototype-output`
- Result: tree API returned 574 items; 10 blobs fetched and saved; raw miss for
  `.devcontainer/devcontainer.json` and `.devcontainer/post-create.sh` recovered via the
  contents API; deliberate missing-file test returned HTTP 404 and was logged gracefully.
- Rate-limit remaining stayed above 55 after 1 tree call + 11 API calls.

- **Tooling baseline:**
- `node --version` = `v25.4.0` (global `fetch` available, ESM supported).
- `npm --version` = `11.13.0`.
- `package.json` has `"type": "module"` and lists `undici` as a dependency, but the
  prototype uses the built-in global `fetch` to avoid extra imports.
- Windows PowerShell is the local shell, so the implementation will be a Node ESM script
  rather than a shell pipeline.

#### Step 03 — Write red tests / fixtures for the assimilation script [DONE]

```yaml
phase: 1
step: 3
title: 'Write red tests / fixtures for the assimilation script'
goal: 'red-testing'
tdd_sequence: 'red-green'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 04 — Implement the skill, agent, templates, and script'
skills:
  - 'red-test-contracts'
  - 'creating-unit-tests'
  - 'execute'
specialists:
  - 'unit-test-writer'
  - 'planning-test-strategy-coordinator'
validation:
  - 'NODE_OPTIONS="--experimental-vm-modules --no-experimental-webstorage" npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation/assimilate-repo.test.ts'
  - 'node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
  - 'node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
acceptance_criteria:
  - 'Test file exists at testing\assimilation\assimilate-repo.test.ts and imports from scripts\assimilation\assimilate-repo.mjs'
  - 'Mocked fetch fixtures simulate GitHub raw miss, contents API hit, tree API response, and 404'
  - 'Tests cover: URL parsing, output folder creation, file download, fetch-failures.md log, license detection, rate-limit reporting'
  - 'Tests fail before the script implementation exists'
  - 'Tests are deterministic and do not depend on live GitHub'
```

**User instruction:** Paste this full step packet.

**Step objective:** Create failing tests that define the expected behavior of the
`assimilate-repo.mjs` script before writing the script.

**Context the agent must know:**

- The script will live at `scripts\assimilation\assimilate-repo.mjs`.
- Tests should live at `scripts\assimilation\assimilate-repo.test.mjs` or `testing\assimilation\...`.
- Use mocked HTTP responses and local fixtures so tests are fast and deterministic.
- If no script implementation is needed (e.g., everything happens inside the agent),
  record an explicit skipped-step packet instead.

**Execution steps:**

1. Decide test location and runner (Jest project).
2. Write tests for:

- URL parsing → owner/repo/ref.
- Repo-name derivation → safe folder name.
- Folder creation with user override.
- Tree fetch → file list.
- File fetch → verbatim save.
- LICENSE detection and recording.

3. Run tests; confirm they fail for the right reason.
4. Update plan with test paths and failure evidence.

**Stop conditions:**

- **Done:** Red tests exist and fail before implementation.
- **Blocked:** Jest project setup is unclear; escalate to `00-helping`.
- **Route-back:** If the scope should be agent-only, route back to Step 01 to skip this step.

**Required validation:**

- `NODE_OPTIONS="--experimental-vm-modules --no-experimental-webstorage" npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation/assimilate-repo.test.ts`
- `node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`

**Plan update requirement:** Record test paths and red-test evidence.

**Step 03 red-test evidence:**

- Test file: `testing\assimilation\assimilate-repo.test.ts`
- Stub file: `scripts\assimilation\assimilate-repo.mjs`
- Focused validation command:

```bash
NODE_OPTIONS="--experimental-vm-modules --no-experimental-webstorage" npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation/assimilate-repo.test.ts
```

- Result: `Test Suites: 1 failed, 1 total; Tests: 10 failed, 10 total`. All failures are honest red-phase failures because the stub functions throw `not implemented`:
- `parseRepoUrl not implemented (url=https://github.com/owner/repo)`
- `parseRepoUrl not implemented (url=https://github.com/owner/repo/tree/main)`
- `deriveFolderName not implemented ...`
- `assimilateRepo not implemented ...`
- Plan-sync gate: pass.
- Step-packet gate: pass.
- Handoff: Step 04 should implement `parseRepoUrl`, `deriveFolderName`, and `assimilateRepo` in `scripts\assimilate-repo.mjs` against these tests.

#### Step 04 — Implement the skill, agent, templates, and script [DONE]

```yaml
phase: 1
step: 4
title: 'Implement the skill, agent, templates, and script'
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 05 — Green validation and sample assimilation'
skills:
  - 'creating-specialist-agent'
  - 'skill-frontmatter-standards'
  - 'agent-frontmatter-standards'
  - 'implementation-standards'
  - 'license-attribution-audit'
  - 'execute'
specialists:
  - 'implementation-executor'
  - 'boundary-mapper'
validation:
  - 'node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md'
  - 'node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts\\assimilation'
  - 'npm run agents:routing-table'
acceptance_criteria:
  - '.github\skills\external-tool-assimilation\SKILL.md exists with valid frontmatter'
  - '.github\agents\assimilator.agent.md exists with valid frontmatter and bounded delegation'
  - '.github\templates\assimilation\*.template.md exist for overview, workflow, validation, and synthesis'
  - 'scripts\assimilation\assimilate-repo.mjs exists and passes red tests'
  - 'Routing table is regenerated and freshness gate passes'
  - 'License-attribution notes are recorded for any external text used in templates'
```

**User instruction:** Paste this full step packet.

**Step objective:** Build the deliverables: skill playbook, specialist agent, templates,
and the repo-fetch script.

**Context the agent must know:**

- Skill/agent templates must follow existing frontmatter schemas.
- The `assimilator` agent is Tier 3, hidden, with `agents: []` unless it needs a T4 summarizer.
- The skill owns the durable process; the agent owns the autonomous execution.
- Templates must not reproduce large verbatim passages from CC-licensed sources.
- ISO 42001 learning events must be recorded for each assimilation run.

**Execution steps:**

1. Create `.github\skills\external-tool-assimilation\SKILL.md`.
2. Create `.github\agents\assimilator.agent.md`.
3. Create template files under `.github\templates\assimilation\`.
4. Implement `scripts\assimilation\assimilate-repo.mjs`.
5. Wire the new skill/agent into parent orchestrators if required (or document the trigger phrase).
6. Regenerate the routing table.
7. Run validators.

**Stop conditions:**

- **Done:** All deliverables exist and pass validators.
- **Blocked:** Frontmatter or graph validation fails; fix and retry.
- **Route-back:** If agent/skill frontmatter or graph validation fails, route back to
  Step 01 or fix in-place; only escalate to `00-helping` for unresolvable cross-tier
  coordination issues.

**Required validation:**

- `node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md`
- `node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict`
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts\\assimilation`
- `npm run agents:routing-table`

**Plan update requirement:** Record deliverable paths and validation evidence.

#### Step 05 — Green validation and sample assimilation [DONE]

```yaml
phase: 1
step: 5
title: 'Green validation and sample assimilation'
goal: 'green-testing'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 06 — Documentation and usage examples'
skills:
  - 'green-validation-gates'
  - 'plan-sync-validation'
  - 'execute'
specialists:
  - 'green-test-failure-triage-coordinator'
  - 'unit-test-runner'
validation:
  - '$env:NODE_OPTIONS="--experimental-vm-modules"; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation'
  - 'node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md'
  - 'node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict'
  - 'npm run agents:routing-table'
  - 'npm run agents:routing-table:gate'
  - 'npm run lint'
  - 'node scripts\agent-customization\validate-agent-graph.mjs --json'
  - 'node scripts\assimilation\assimilate-repo.mjs https://github.com/github/choosealicense.com C:\NeatapticTS\tmp\green-sample-choosealicense'
  - 'node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
  - 'node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
  - 'neataptic-gate-mcp:run_gate_check --gate=agent-graph'
  - 'neataptic-gate-mcp:run_gate_check --gate=routing-table-freshness'
acceptance_criteria:
  - 'All assimilation script tests pass'
  - 'Agent and skill frontmatter validators pass'
  - 'Routing table is fresh'
  - 'Lint passes (or pre-existing failures are triaged and fixed)'
  - 'Agent-graph gate passes'
  - 'Plan-sync and step-packet gates pass'
  - 'Sample end-to-end assimilation of a small public repo completes and produces the expected artifacts'
```

**User instruction:** Paste this full step packet.

**Step objective:** Validate the new skill/agent/script against the red tests, routing
table gates, and a live sample repo.

**Context the agent must know:**

- Use targeted tests only; do not run the full NeatapticTS test suite unless explicitly approved.
- The sample repo should be small and permissively licensed to keep validation fast.
- If GitHub is unavailable, use a local mock repo tarball.

**Execution steps:**

1. Run targeted assimilation tests.
2. Run routing-table and agent-graph gates.
3. Run a sample `/assimilate https://github.com/<small-repo>` end-to-end.
4. Inspect the generated folder for the expected structure.
5. Record validation evidence.

**Stop conditions:**

- **Done:** All gates pass and the sample run produces the expected artifacts.
- **Blocked:** A gate fails; route back to the smallest relevant step.
- **Route-back:** If the sample reveals process gaps, route back to Step 01/04.

**Required validation:**

- `$env:NODE_OPTIONS="--experimental-vm-modules"; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation`
- `node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md`
- `node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict`
- `npm run agents:routing-table`
- `npm run agents:routing-table:gate`
- `npm run lint`
- `node scripts\agent-customization\validate-agent-graph.mjs --json`
- `node scripts\assimilation\assimilate-repo.mjs https://github.com/github/choosealicense.com C:\NeatapticTS\tmp\green-sample-choosealicense`
- `node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync`
- `neataptic-gate-mcp:run_gate_check --gate=step-packet`
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph`
- `neataptic-gate-mcp:run_gate_check --gate=routing-table-freshness`

**Plan update requirement:** Record green validation evidence.

##### Step 05 validation evidence

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation` — **PASS** (1 suite, 10/10 tests)
- `node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md` — **PASS** (ok=true)
- `node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict` — **PASS** (0 errors, 0 warnings)
- `npm run agents:routing-table` — **PASS** (changed=false, agents=66, skills=61)
- `npm run agents:routing-table:gate` — **PASS**
- `npm run lint` — **PASS**
- `node scripts\agent-customization\validate-agent-graph.mjs --json` — **PASS** (0 errors, 0 warnings)
- Sample end-to-end assimilation of `github/choosealicense.com` — **PASS** (149 files downloaded, 0 failures)
- Verified sample artifacts: `tmp\green-sample-choosealicense\choosealicense.com\verbatim\` (122 files), `repo.md`, `notes\license-attribution.md`, `notes\fetch-failures.md`
- `node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md` — **PASS**
- `node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=step-packet` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=routing-table-freshness` — **PASS**
- Recorded gate exceptions during the first validation attempt (CLI entry-point guard and synthesis filename); both fixed via `04-implementing` slice-fixes and re-validated above.
- `04-implementing` also cleaned up the `tmp\green-sample-choosealicense` sample folder.

#### Step 06 — Documentation and usage examples [DONE]

```yaml
phase: 1
step: 6
title: 'Documentation and usage examples'
goal: 'documenting'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'Step 07 — Session logging and tracker closure'
skills:
  - 'educational-docs'
  - 'license-attribution-audit'
  - 'execute'
specialists:
  - 'docs-example-writer'
validation:
  - 'npm run docs:quality:metrics -- --selectors=assimilation' # CLI does not support this flag; ran manual Markdown review instead
  - 'Manual read-through of .github\skills\external-tool-assimilation\SKILL.md'
  - 'Manual read-through of .github\agents\assimilator.agent.md'
acceptance_criteria:
  - 'Skill README explains the /assimilate trigger and the end-to-end flow'
  - 'Agent description and argument-hint are clear and triggerable'
  - 'Usage example shows the generated folder layout'
  - 'License-attribution section is present and accurate'
```

**Step 06 validation evidence**

- `npx prettier --check` / `npx prettier --write` on edited `.md` files — **PASS**
- `node scripts\agent-customization\validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md` — **PASS** (ok=true, 5 pre-existing tool-alias warnings)
- `node scripts\agent-customization\validate-skill-frontmatter.mjs --json --strict` — **PASS** (0 errors, 0 warnings)
- `node scripts\agent-customization\validate-agent-quality.mjs --json` — **PASS** (0 errors, 0 warnings after adding `## Output format` block)
- `node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md` — **PASS**
- `node scripts\agent-customization\validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md` — **PASS**
- `npm run agents:routing-table` — **PASS** (regenerated table with updated assimilator content)
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=step-packet` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=routing-table-freshness` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=agent-quality` — **PASS**
- `neataptic-gate-mcp:run_gate_check --gate=learning-event` — **PASS**
- `npm run docs:quality:metrics -- --selectors=assimilation` — **NOT APPLICABLE** (script does not support `--selectors`; it scans `src/**/*.ts` only). Performed manual review of SKILL.md and agent file instead.
- `node rag-index/build-index.mjs` — **PASS** (indexed changed docs surfaces)
- `neataptic-gate-mcp:run_gate_check --gate=cortex-first-search` — **PASS** after rebuild
- `neataptic-gate-mcp:run_gate_check --gate=cortex-index` — **FAIL** (`workflow_mcp_alive: false`) due to stale workflow-MCP binding, not a documentation content issue.

**Files changed**

- `.github\skills\external-tool-assimilation\SKILL.md` — added `## Worked example` with trigger, optional target folder, and generated folder layout.
- `.github\agents\assimilator.agent.md` — added `## How to invoke` and corrected `## Output format` to satisfy the agent-quality contract.
- `.github\agent-skill-routing-table.md` — regenerated.
- `spec-kit\notes\license-attribution.md` — created (MIT attribution for canonical example).
- `spec-kit\verbatim\LICENSE` — added original `github/spec-kit` LICENSE.
- `.github\ai-learning\learning-log.jsonl` — appended `external-tool-assimilated` ISO 42001 learning event.

**Handoff to Step 07**

Step 06 is complete. The `/assimilate` trigger and the `assimilator` agent are
now documented with a worked example and a correct output contract. The
remaining environment-side gate failure (`cortex-index` workflow MCP not alive)
should be resolved by restarting the workflow MCP before Step 07 begins.

#### Step 07 — Session logging and tracker closure [PLANNED]

```yaml
phase: 1
step: 7
title: 'Session logging and tracker closure'
goal: 'logging'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans\completed\assimilate-repeatable-skill.plans.md'
copy_paste: true
next_step: 'null — plan/log pair archived to plans\completed\'
skills:
  - 'tracker-handoff'
  - 'capturing-learning-event'
  - 'plan-sync-validation'
  - 'execute'
specialists:
  - 'learning-event-capturer'
  - 'file-change-summarizer'
validation:
  - 'node scripts\agent-customization\gates\phase-compression.gate.mjs --json'
  - 'node scripts\agent-customization\gates\log-completion-marker.gate.mjs --json'
  - 'node scripts\agent-customization\gates\stale-wip-plans.gate.mjs --json'
  - 'node scripts\agent-customization\validate-plan-sync.mjs --json --plan=plans\completed\assimilate-repeatable-skill.plans.md'
acceptance_criteria:
  - 'Phase 1 history is compressed to a concise coverage note'
  - 'Matching .logs.md file exists with durable done-state record'
  - 'Plan and log are moved to plans\completed\'
  - 'Learning event recorded for the new skill/agent creation'
  - 'stale-wip-plans gate passes'
```

**User instruction:** Paste this full step packet.

**Step objective:** Compress the completed phase, create the audit log, move the closed
plan pair to `plans\completed\`, and record an ISO 42001 learning event.

**Context the agent must know:**

- Follow `tracker-handoff` compression rules.
- Do not leave a stale `Handoff query` on a closed plan.
- Record the event in `.github\ai-learning\learning-log.jsonl`.

**Execution steps:**

1. Compress Phase 1 step details into a `[DONE]` coverage note.
2. Create `plans\assimilate-repeatable-skill.logs.md`.
3. Move both files to `plans\completed\`.
4. Record a `skill-update`/`agent-update` learning event.
5. Run closure gates.

**Stop conditions:**

- **Done:** Plan pair archived, gates pass, learning event recorded.
- **Blocked:** A closure gate fails; fix and retry.

**Required validation:**

- `node scripts\agent-customization\gates\phase-compression.gate.mjs --json`
- `node scripts\agent-customization\gates\log-completion-marker.gate.mjs --json`
- `node scripts\agent-customization\gates\stale-wip-plans.gate.mjs --json`

**Plan update requirement:** Final status update before archive.

---

## Validation gates

This plan uses plan-sync, step-packet, agent-graph, frontmatter, routing-table,
and targeted Jest validation. See the per-step validation lists above and the
**Latest validation evidence** section at the end of this file.

## Deliverable files

| File or folder                                             | Purpose                                                                  | Created in step |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ | --------------- |
| `.github\skills\external-tool-assimilation\SKILL.md`       | Durable workflow: trigger, process, comparison dimensions, license rules | 04              |
| `.github\agents\assimilator.agent.md`                      | Hidden Tier 3 specialist that runs the workflow                          | 04              |
| `.github\templates\assimilation\00-overview.template.md`   | Template for the high-level overview summary                             | 04              |
| `.github\templates\assimilation\01-workflow.template.md`   | Template for workflow/commands analysis                                  | 04              |
| `.github\templates\assimilation\02-validation.template.md` | Template for validation/gate analysis                                    | 04              |
| `.github\templates\assimilation\03-artifacts.template.md`  | Template for templates/artifact analysis                                 | 04              |
| `.github\templates\assimilation\04-ecosystem.template.md`  | Template for extensions/ecosystem analysis                               | 04              |
| `.github\templates\assimilation\05-learning.template.md`   | Template for ISO/learning-event overlap analysis                         | 04              |
| `.github\templates\assimilation\99-synthesis.template.md`  | Template for the final condensed `<folder>.md`                           | 04              |
| `scripts\assimilation\assimilate-repo.mjs`                 | ESM script that fetches tree/files and scaffolds the folder              | 04              |
| `scripts\assimilation\assimilate-repo.test.mjs`            | Deterministic unit tests with fixtures/mocks                             | 03              |
| `plans\assimilate-repeatable-skill.logs.md`                | Durable done-state record after closure                                  | 07              |
| `.github\agent-skill-routing-table.md`                     | Regenerated after adding the new agent/skill                             | 04              |

---

## Step-by-step assimilation process

The skill/agent must reproduce the spec-kit manual process automatically:

1. **Trigger parse**

- Accept `/assimilate <github-url> [<target-folder>]`.
- Derive owner/repo/ref from the URL.
- Derive folder name from repo name if not supplied.
- Validate the folder does not already exist unless `--force` is given.

2. **Repo discovery**

- Fetch `README.md` and `LICENSE` first.
- Fetch the recursive tree via GitHub API to inventory structure.
- Identify meaningful surfaces:
- top-level docs,
- core concepts / philosophy,
- commands / workflows,
- validation / quality gates,
- templates / artifacts,
- extensions / ecosystem,
- learning / disclosure patterns.

3. **Deep file retrieval**

- For each meaningful surface, fetch the relevant files using raw GitHub first,
  API `contents` endpoint second, and git-clone third.
- Record any fetch failures in `<folder>\notes\fetch-failures.md`.

4. **Verbatim persistence**

- Save important source files under `<folder>\verbatim\` preserving relative paths.
- Do not edit verbatim copies.

5. **Local NeatapticTS inventory**

- Read the local agent/skill/routing table, plan index, and roadmap.
- Identify comparable surfaces in NeatapticTS for each external surface.

6. **Per-area summaries**

- Generate `00-overview.md`, `01-workflow.md`, `02-validation.md`,
  `03-artifacts.md`, `04-ecosystem.md`, `05-learning.md` using the templates.
- Each summary compares the external tool with NeatapticTS and flags gaps.

7. **Final synthesis**

- Generate `<folder>.md` with:
- executive summary,
- top-N cherry-pick recommendations,
- area-by-area comparison table,
- what to preserve vs. what to adopt,
- concrete next steps,
- reference map.

8. **License verification**

- Detect `LICENSE` type and obligations.
- Record attribution in `<folder>\notes\license.md` and in the skill references file.
- If license is unknown or restrictive, block further use and escalate.

---

## Comparison criteria and summary file naming convention

| File                     | Dimension                                              | Compared with NeatapticTS                                        |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------- |
| `00-overview.md`         | What the tool is, philosophy, positioning              | Local agents/skills, plan trackers, README                       |
| `01-workflow.md`         | Commands, SDLC loop, artifact lifecycle                | `phase-handoff-workflow`, step-packet YAML                       |
| `02-validation.md`       | Quality gates, checklists, analysis cycles             | `green-validation-gates`, `plan-sync-validation`, gate contracts |
| `03-artifacts.md`        | Templates, traceability IDs, file layouts              | `plans/*.plans.md`, `.github/templates/`                         |
| `04-ecosystem.md`        | Extensions, presets, bundles, catalog patterns         | `.github/skills/`, `.github/agents/`, routing table              |
| `05-learning.md`         | Disclosure, learning events, ISO 42001                 | `capturing-learning-event`, `learning-log.jsonl`                 |
| `06-verbatim-phrases.md` | Exact phrasing, verbs, section headings worth adopting | Local agent/skill descriptions                                   |
| `<folder>.md`            | Condensed synthesis and recommendations                | —                                                                |

File numbering is two digits so sorts lexicographically. Verbatim source files live
under `verbatim\` with their original relative paths preserved.

---

## Validation approach

### Unit/script level

- Deterministic Jest tests for `scripts\assimilation\assimilate-repo.mjs` using
  mocked fetch fixtures.
- Test cases cover URL parse, folder naming, tree fetch, file fetch, LICENSE detection,
  and failure logging.

### Customization level

- `validate-agent-frontmatter.mjs --json --agent .github\agents\assimilator.agent.md`
- `validate-skill-frontmatter.mjs --json --strict`
- `validate-agent-graph.mjs --json`
- `npm run agents:routing-table` + `npm run agents:routing-table:gate`

### End-to-end level

- Run `/assimilate https://github.com/github/spec-kit` (or another permissively
  licensed small repo) and verify the output folder matches the manual spec-kit
  structure in shape and content.

### Plan-sync level

- `validate-plan-sync.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`
- `validate-plan-phase-packets.mjs --json --plan=plans\assimilate-repeatable-skill.plans.md`

---

## ISO 42001 learning-event notes

Every assimilation run must append a structured event to `.github\ai-learning\learning-log.jsonl`.
The skill should define a new event type `external-tool-assimilated` with these fields:

```json
{
  "eventType": "external-tool-assimilated",
  "triggeringTask": "/assimilate <repo-url>",
  "externalRepo": "<owner>/<repo>",
  "license": "<SPDX identifier or detected name>",
  "localFolder": "<folder>",
  "filesChanged": ["<folder>\\00-overview.md", "<folder>\\verbatim\\..."],
  "agentsAffected": ["assimilator"],
  "skillsAffected": ["external-tool-assimilation"],
  "confirmation": "user-confirmed|not-required",
  "resumeAction": "Review <folder>.md synthesis and decide which recommendations to implement"
}
```

The creation of the skill and agent themselves is a `skill-update` + `agent-update`
event recorded during Step 07.

---

### Latest validation evidence

Phase 1 completed. Detailed evidence moved to plans\completed\assimilate-repeatable-skill.logs.md.
