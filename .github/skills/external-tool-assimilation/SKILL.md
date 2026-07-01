---
name: external-tool-assimilation
description: 'Use when: a Tier-1 orchestrator detects `/assimilate <github-url>` and needs to study an external GitHub repository, persist its important surfaces, compare them with NeatapticTS, and produce actionable cherry-pick recommendations.'
argument-hint: 'Provide the public GitHub repository URL, optional target folder name, and any comparison dimensions the orchestrator wants emphasized.'
user-invocable: false
disable-model-invocation: false
tier: 2
skills:
  - license-attribution-audit
  - capturing-learning-event
  - research-methodology
  - plan-alignment
  - tracker-handoff
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# External Tool Assimilation Playbook

Use this skill when the NeatapticTS customization system needs to absorb useful
practices, phrasing, or structures from an external open-source GitHub repository
in a repeatable, license-respecting way.

This skill owns the durable workflow for the `/assimilate <github-url>` trigger:
repo discovery, verbatim extraction, per-area analysis against NeatapticTS,
synthesis of cherry-pick recommendations, license attribution, and ISO 42001
learning-event recording.

## When to Use

- A user or orchestrator asks to study an external tool and compare it with NeatapticTS.
- A new skill/agent request clearly maps to a pattern already solved elsewhere.
- A plan, gate, or agent description could benefit from proven phrasing or structure
  from a permissively licensed external repo.
- A research phase needs persisted verbatim sources plus a local comparison summary.

## When NOT to Use

Do NOT use for importing code into `src/` without a separate implementation plan and
coverage gate. Do NOT use for repositories with unknown or restrictive licenses.
Do NOT use when a simple web search would suffice.

## Trigger

`/assimilate <github-url> [target-folder]`

The phrase is detected by `01-planning` or `02-researching`; they delegate the actual
analysis to the hidden Tier-3 `assimilator` specialist agent.

## Workflow Diagram

```text
Flowchart summary: "Trigger /assimilate" → "Parse URL + derive folder"; "Parse URL + derive folder" → "Fetch README + LICENSE"; "Fetch README + LICENSE" → "Fetch recursive tree"; "Fetch recursive tree" → "Download blobs (raw → contents → git clone fallback)"; "Download blobs (raw → contents → git clone fallback)" → "Write verbatim/"; "Write verbatim/" → "Compare each surface with NeatapticTS"; "Compare each surface with NeatapticTS" → "Fill per-area templates"; "Fill per-area templates" → "Write final synthesis"; "Write final synthesis" → "Record license + learning event"; "Record license + learning event" → "Return bounded report".
```

## Task Packet

Pass a compact packet to the `assimilator` agent:

```text
Use external-tool-assimilation for /assimilate https://github.com/owner/repo.
Target folder: <optional>.
Focus: <overview | workflow | validation | templates | ecosystem | learning | verbatim>.
Expected output: per-area summaries + <folder>.md synthesis + ISO 42001 learning event.
```

## Required Workflow

1. **Parse the trigger.** Extract `owner`, `repo`, `ref` from the GitHub URL.
2. **Derive the workspace folder.** Default to the repo name under the project root
   or a user-supplied parent folder.
3. **Fetch high-signal files first.** Read `README.md` and `LICENSE` before the tree
   so the agent can understand positioning and obligations early.
4. **Fetch the recursive tree.** Use `https://api.github.com/repos/<owner>/<repo>/git/trees/<ref>?recursive=1`.
5. **Download each blob using the fetch ladder:**
   - `https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>` (primary)
   - `https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<ref>` (fallback,
     base64-decode the `content` field)
   - `git clone --depth 1 <repo-url> <folder>` (final fallback when API limits or
     unusual content prevent individual fetches)
6. **Record verbatim copies** under `<folder>/verbatim/` preserving relative paths.
7. **Log failures** in `<folder>/notes/fetch-failures.md`.
8. **Detect and record the license** in `<folder>/notes/license-attribution.md`.
9. **Compare surfaces** with the local NeatapticTS inventory:
   - agents/skills/routing table,
   - plan trackers and step packets,
   - gate contracts,
   - ISO 42001 learning-log conventions.
10. **Fill per-area templates** under `<folder>/`:
    - `00-overview.md`
    - `01-sdd-workflow.md`
    - `02-validation-cycles.md`
    - `03-planning-triage.md`
    - `04-templates-artifacts.md`
    - `05-extensions-ecosystem.md`
    - `06-iso42001-learning-overlap.md`
    - `07-verbatim-phrases-and-verbs.md`
11. **Write the final synthesis** `<folder>.md` with:
    - executive summary,
    - top-N cherry-pick recommendations,
    - area-by-area comparison table,
    - what to preserve vs. adopt,
    - concrete next steps,
    - reference map.
12. **Record an ISO 42001 learning event** of type `external-tool-assimilated` in
    `.github/ai-learning/learning-log.jsonl`.
13. **Return a bounded report** to the orchestrator with paths, license, and recommendations.

## Comparison Dimensions

| File                               | Dimension                                              | Compared with NeatapticTS                                        |
| ---------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| `00-overview.md`                   | What the tool is, philosophy, positioning              | Local agents/skills, plan trackers, README                       |
| `01-sdd-workflow.md`               | Commands, SDLC loop, artifact lifecycle                | `phase-handoff-workflow`, step-packet YAML                       |
| `02-validation-cycles.md`          | Quality gates, checklists, analysis cycles             | `green-validation-gates`, `plan-sync-validation`, gate contracts |
| `03-planning-triage.md`            | Planning, triage, ambiguity handling                   | `01-planning`, `planning-risk-coordinator`                       |
| `04-templates-artifacts.md`        | Templates, traceability IDs, file layouts              | `plans/*.plans.md`, `.github/templates/`                         |
| `05-extensions-ecosystem.md`       | Extensions, presets, bundles, catalog patterns         | `.github/skills/`, `.github/agents/`, routing table              |
| `06-iso42001-learning-overlap.md`  | Disclosure, learning events, ISO 42001                 | `capturing-learning-event`, `learning-log.jsonl`                 |
| `07-verbatim-phrases-and-verbs.md` | Exact phrasing, verbs, section headings worth adopting | Local agent/skill descriptions                                   |
| `<folder>.md`                      | Condensed synthesis and recommendations                | —                                                                |

## Fetch Ladder

1. `raw.githubusercontent.com` — fast, high rate-limit, no JSON decoding.
2. GitHub contents API — fallback for raw misses; decode `content` from base64.
3. GitHub tree API — enumerate the full file tree once per repo.
4. `git clone --depth 1` — final fallback for API exhaustion, LFS, or submodules.

## Rate-Limit and 404 Rules

- Inspect `x-ratelimit-remaining` and `x-ratelimit-reset` on every API response.
- Support an optional `GITHUB_TOKEN` for 5,000 requests/hour.
- On raw 404/403/5xx, fall back to the contents API immediately.
- On contents API 404, record the path and continue.
- Use bounded concurrency (default 5 parallel fetches) and exponential back-off.

## License and Attribution Rules

- Always fetch the repo `LICENSE` file first.
- Detect the license type and record obligations in `<folder>/notes/license-attribution.md`.
- For unknown or restrictive licenses, stop and escalate before copying any verbatim text.
- Keep copied verbatim excerpts short and include source attribution.
- Never reproduce long passages from external sources inside NeatapticTS skills or agents.

## ISO 42001 Learning Event

Every assimilation run must append a structured event to
`.github/ai-learning/learning-log.jsonl`:

```json
{
  "timestamp": "<ISO timestamp>",
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

## Guardrails

- Do not let external assumptions leak into NeatapticTS agent/skill semantics.
- Do not copy long verbatim passages into templates or skills.
- Do not claim compatibility with an external tool beyond what the comparison proves.
- Do not skip the license check.
- Do not leave fetch failures unlogged.
- Do not bypass the ISO 42001 learning-event step.

## Expected Final Output

A strong assimilation pass should report:

- the external repo owner/name/ref,
- the local folder and its layout,
- the detected license and attribution path,
- the number of files downloaded and failed,
- the per-area summary files generated,
- the cherry-pick recommendations in `<folder>.md`,
- the ISO 42001 learning event recorded.

## Worked example

Suppose you want to study how GitHub documents software licenses so you can
compare its conventions with the NeatapticTS license-attribution workflow.

The trigger phrase is:

```text
/assimilate https://github.com/github/choosealicense.com
```

A Tier-1 orchestrator such as `01-planning` or `02-researching` detects the
phrase, parses the URL, and delegates the actual work to the hidden Tier-3
`assimilator` specialist using the `task` tool. You do not invoke `assimilator`
directly.

If you want the output placed somewhere other than a repo-named folder under the
project root, add a second argument:

```text
/assimilate https://github.com/github/choosealicense.com docs/research/licenses
```

After the run finishes, the workspace contains a verbatim copy of the external
repo, per-area comparison notes, and a condensed synthesis. The exact layout for
`choosealicense.com` looks like this:

```text
choosealicense.com/
├── verbatim/
│   ├── README.md
│   ├── LICENSE.md
│   ├── _data/
│   │   └── licenses.yml
│   └── ...        # remaining downloaded files preserved in place
├── notes/
│   ├── license-attribution.md
│   └── fetch-failures.md
├── 00-overview.md
├── 01-sdd-workflow.md
├── 02-validation-cycles.md
├── 03-planning-triage.md
├── 04-templates-artifacts.md
├── 05-extensions-ecosystem.md
├── 06-iso42001-learning-overlap.md
├── 07-verbatim-phrases-and-verbs.md
└── choosealicense.com.md
```

The `verbatim/` folder is the raw evidence shelf. The `notes/` folder records
the detected license and any files that could not be downloaded. The
`00-` through `07-` files compare specific surfaces with their NeatapticTS
counterparts, and `choosealicense.com.md` is the one-page synthesis the
orchestrator reads first when deciding which recommendations to act on.

This same pattern works for any public GitHub repository: pass the URL,
optionally name a target folder, and receive a bounded study that can be
reviewed before any local source file is changed.
