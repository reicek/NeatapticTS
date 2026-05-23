```structured-v1
OUTPUT_CONTRACT: structured-v1
TASK_STATUS: SUCCESS
TIER: 0
ROLE: 04-implementing
TASK_RECEIVED: Apply the smallest safe fix for the output contract gap.
FILES_READ:
- .github/agents/04-implementing.agent.md
- scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs
FILES_CHANGED:
- .github/agents/04-implementing.agent.md
- scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs
KEY_FINDINGS:
- Tier-0 prompts previously documented only prose output requirements.
ACTIONS_TAKEN:
- Added the exact Tier-0 structured-v1 prompt contract.
VALIDATION_EVIDENCE:
- node scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs --contract=tier0 --input=scripts/agent-customization/fixtures/numbered-agent-structured-v1.valid.md
BLOCKERS:
- NONE
RISKS_OR_GAPS:
- NONE
LEARNING_EVENT_NEEDED: false
SUGGESTED_NEXT_AGENT: 05-green-testing
PHASE_COMPLETE: true
SUB_ORCHESTRATORS_USED: none
SUMMARY: Minimal passing Tier-0 structured-v1 fixture for exact-contract validation.
```