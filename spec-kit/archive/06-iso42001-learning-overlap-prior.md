# Spec Kit — ISO 42001 / Learning Event Overlap

Spec Kit and NeatapticTS both recognize that AI-assisted workflows need transparency, but they address the problem differently. Spec Kit focuses on **human-facing disclosure** in PRs, commits, and comments. NeatapticTS focuses on **machine-readable learning events** that create a durable, auditable record of system changes. This section compares the two approaches and notes what NeatapticTS should preserve.

## Spec Kit: Disclosure and Attestation

Spec Kit's `AGENTS.md` (which is active as a loaded custom instruction after this study) contains a section titled **Agent Disclosure for PRs, Comments, and Commits**.

### Continuous Disclosure

> "Disclosure is **continuous**, not a one-time event. A single AI-disclosure paragraph in the PR body does **not** cover the commits and replies you add during review rounds. Each of the following must independently attest to agent authorship."

This principle is stronger than a one-time checkbox; it requires every commit and every review-round comment to disclose agent involvement.

### Commit Trailer

> "Every commit you authormust carry an `Assisted-by:` trailer** identifying the agent and whether it acted autonomously or under direct human supervision"

Example:

```text
Assisted-by: GitHub Copilot (model: <name-if-known>, autonomous)
```

The distinction between `autonomous` and `supervised` is meaningful: only `supervised` is allowed when a human authored or line-by-line reviewed the change.

### Comment Disclosure

For PR comments and review replies:

> "If you are an agent working on behalf of a human, **disclose your identity in your PR comment** — name the agent (and model, if applicable) and the human you are acting for"
> "Re-state agent identity in each review-round summary comment."

### Anti-Patterns

`AGENTS.md` explicitly forbids speed-of-turnaround as a substitute for disclosure:

> "Do not reply 'Done' or push a 'fix' within seconds/minutes of a review event without disclosing that the response or commit was agent-generated."

This is a governance posture: automated responses must be labeled as automated.

### What Spec Kit Lacks

There is **no explicit structured event schema** for recording *why* the workflow changed. Disclosure is about authorship and autonomy, not about system improvement. Spec Kit does not define:

- A machine-readable log of routing updates, skill updates, model changes, or contract fixes
- Event types or a schema
- An append-only audit file
- A "resume action" field for session continuity

The workflow transparency is high for *individual commits and comments*, but low for *system-level learning*.

---

## NeatapticTS: `capturing-learning-event` Skill

NeatapticTS has a dedicated skill and a supporting flow for ISO-42001-style learning evidence.

### Purpose

> "This skill appends a structured, append-only learning evidence record to `.github/ai-learning/learning-log.jsonl`. It documents agent-system gaps, corrective updates, and routing changes in a compact, public-project-friendly format that supports session continuity without making compliance claims."

The skill is careful not to overclaim:

> "Do not claim ISO-42001 certification or compliance; this is a local evidence log, not a certified system."

### Event Types

The schema defines five event types:

- `agent-system-gap`
- `agent-update`
- `skill-update`
- `routing-update`
- `output-contract-fix`

These map exactly to the kinds of friction an agent-based SDLC experiences.

### Schema

```json
{
  "timestamp": "<ISO timestamp>",
  "eventType": "agent-system-gap|agent-update|skill-update|routing-update|output-contract-fix",
  "triggeringTask": "<brief>",
  "gap": "<what was missing>",
  "resolution": "<what changed>",
  "filesChanged": ["<path>"],
  "agentsAffected": ["<agent-name>"],
  "skillsAffected": ["<skill-name>"],
  "confirmation": "not-required|user-confirmed|deferred",
  "resumeAction": "<how work continued>"
}
```

### Append-Only Discipline

> "Do not delete or rewrite existing log entries; always append."

This makes the log an immutable-ish evidence stream.

### Flow Support

A dedicated flow exists (`07.learning-event-log.flow.yml`) and a hidden specialist agent (`learning-event-capturer.agent.md`) can be dispatched to record events. The skill is wired into the routing table and is invoked when workflow gaps are discovered.

---

## Side-by-Side Comparison

| Concern | Spec Kit | NeatapticTS |
|---------|----------|-------------|
| **Commit attribution** | `Assisted-by:` trailer required on every commit | Uses `Co-authored-by:` for Copilot-generated commits; no `Assisted-by:` requirement |
| **Autonomy label** | `autonomous` vs `supervised` per commit | Not formalized per commit |
| **PR/review disclosure** | Continuous disclosure per comment/round | No explicit skill for comment disclosure |
| **System-level learning log** | None | `learning-log.jsonl` with typed events |
| **Event schema** | None | JSONL schema with timestamp, type, gap, resolution, files, resume action |
| **Audit purpose** | Authorship attestation | Workflow improvement + session continuity |
| **ISO 42001 framing** | Implied through transparency | Explicit skill name and schema, but guarded against certification claims |
| **Enforcement** | Loaded as custom instruction; relies on agent self-policing | Skill + flow + optional gate / `00.workflow-gap-audit` |

## What NeatapticTS Should Preserve

1. **The `capturing-learning-event` skill and `learning-log.jsonl` schema**. This is the most durable, machine-actionable transparency artifact NeatapticTS has. Spec Kit has nothing equivalent.
2. **Typed event vocabulary** (`agent-system-gap`, `routing-update`, `skill-update`, `output-contract-fix`, `agent-update`). These event types capture the actual failure modes of an agent system.
3. **Append-only discipline**. Rewriting the log would destroy its evidentiary value.
4. **The "resume action" field**. This is critical for session continuity and is absent from Spec Kit.
5. **The guarded ISO framing**. Avoid claiming certification while still producing evidence-quality records.

## What NeatapticTS Could Adopt from Spec Kit

1. **`Assisted-by:` trailer on commits**, with `autonomous`/`supervised` distinction. This is a stronger human-readable attestation than `Co-authored-by:` alone.
2. **Continuous disclosure in PR comments and review rounds**. The current NeatapticTS system does not have a skill or rule for this.
3. **A per-commit / per-comment disclosure policy** loaded as a custom instruction, similar to `AGENTS.md`, so disclosure is automatic rather than opt-in.

## Recommended Synthesis

- Keep the **NeatapticTS learning-event log as the system-of-record** for workflow changes.
- Add an **agent-disclosure custom instruction** that requires `Assisted-by:` trailers and review-round disclosure.
- Consider recording **disclosure-related learning events** (e.g., a `disclosure-policy-update` event type) when the disclosure rules themselves change, so both the authorship trail and the system-evolution trail are covered.
