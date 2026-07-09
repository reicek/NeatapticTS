---
name: capturing-learning-event
description: 'Use when: capturing an ISO-42001-style local AI system learning event.'
argument-hint: 'Describe the gap, change, or fix; the evidence that supports it; the agent source; and whether the event is open or resolved.'
user-invocable: false
disable-model-invocation: false
skills:
  - tracker-handoff
  - agent-frontmatter-standards
  - routing-optimization-policy
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`, `expand_query`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Capturing Learning Events

Use this skill when an agent-system gap, routing update, skill update, model update, or output-contract fix has been identified and should be recorded as durable evidence in the local ISO-42001-style learning log.

This skill is the companion to the `learning-event-capturer` specialist agent. The skill owns the playbook; the agent owns the file edit when invoked by a caller.

## When To Use

- A recurring workflow friction is discovered and should be recorded so future sessions inherit the lesson.
- An agent/skill routing table change needs durable justification.
- A model routing or budget decision changes and the rationale should persist.
- A skill or agent frontmatter fix closes a gap that could recur.
- An output-contract correction is made and the cause should be logged.
- A plan closure, tracker handoff, or post-mortem surfaces a reusable lesson.

## When NOT to use

Do NOT use for transient chat notes or scratch memory. Do NOT use for active tracker management — use `tracker-handoff` instead. Do NOT use for session summaries — use `summarizing-session-log` instead.

## Workflow Diagram

```text
Flowchart summary: "Gap or change identified" → "Confirm evidence"; "Confirm evidence" → "Record smallest learning-event entry"; "Record smallest learning-event entry" → "Run learning-event gate"; "Run learning-event gate" → "Done".
```

## Task Packet

Pass a compact packet that names the event category, the gap or change, the evidence, and the follow-up action.

```text
Use capturing-learning-event for <category>.
Category: agent-system-gap | routing-update | skill-update | model-update | output-contract-fix
Description: <one-line statement of the gap or change>
Evidence: <concrete observation, file path, gate output, or test result>
Follow-up action: <recommended next step or resolution>
Agent source: <agent that identified the gap>
Status: open | resolved
```

## Required Workflow

1. Confirm the gap or change is real and evidence-backed, not a transient confusion.
2. Choose the correct category from the supported set.
3. Write the smallest learning-event record that will prevent rediscovery of the same gap.
4. Append or update the record in `.github/ai-learning/learning-log.jsonl` only.
5. Run the `learning-event` gate to confirm the log entry is well-formed.
6. Cross-reference the event in any related tracker or session summary.

## Learning Event Record Template

```json
{
  "category": "agent-system-gap|routing-update|skill-update|model-update|output-contract-fix",
  "description": "<concise description of the gap or change>",
  "evidence": "<evidence supporting the event>",
  "followup_action": "<recommended follow-up action>",
  "agent_source": "<agent that identified the gap>",
  "status": "open|resolved"
}
```

## Categories

- **agent-system-gap**: A workflow, routing, or agent capability gap that caused a failure or manual workaround.
- **routing-update**: A change to the agent/skill routing table or dispatch policy.
- **skill-update**: A change to a skill definition, frontmatter, or playbook.
- **model-update**: A change to model routing, budget, or agent model assignment.
- **output-contract-fix**: A correction to a structured output contract or phase handoff format.

## Automation And Validation

After recording a learning event, run:

```bash
node scripts/agent-customization/gates/learning-event.gate.mjs --json
```

Record the gate result as evidence. Do not paste the full JSON payload into trackers unless raw output preservation is explicitly requested.

## Guardrails

- Only edit `.github/ai-learning/learning-log.jsonl`.
- Do not make unrelated file edits while recording a learning event.
- Keep each record concise; full transcripts belong in trackers or logs, not the learning log.
- Do not record speculation. Every event must have concrete evidence.
- Do not treat the learning log as a TODO list; it is an evidence record.
- When a recorded gap is later resolved, update the same record's `status` to `resolved` with a brief resolution note rather than creating a duplicate entry.

## Expected Final Output

A learning-event capture should report:

- the event category,
- a one-line description of the gap or change,
- the concrete evidence,
- the recommended follow-up action,
- the agent source,
- the status (`open` or `resolved`),
- whether the `learning-event` gate passed.
