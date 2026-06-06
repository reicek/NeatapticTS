{
  "mission": "Route all substantive work to the smallest relevant numbered SDLC orchestrator. The main Copilot agent must not perform implementation, refactoring, research, planning, testing, documentation, or logging directly.",
  "routing_policy": {
    "exclusive_targets": [
      "00-helping",
      "01-planning",
      "02-researching",
      "03-red-testing",
      "04-implementing",
      "05-green-testing",
      "06-documenting",
      "07-logging"
    ],
    "rules": [
      "Identify the smallest relevant orchestrator for each request before any action.",
      "Delegate immediately; do not begin substantive work before routing.",
      "For whole-plan execution, call 01-planning first, then dispatch remaining orchestrators in order, waiting for completion before advancing.",
      "Exceptions: trivial factual answers (single-sentence lookups, no file changes) and direct operator commands with zero file reads/writes."
    ]
  },
  "flow_and_gate_protocol": {
    "flow_selection": "Numbered SDLC agents select a named flow from .github/flows/ to execute body work.",
    "gate_handling": "Each flow declares exit gates that must return {pass: true, evidence, fixHint, owner} JSON before completion.",
    "post_phase_fanout": "Runs after flow body completes.",
    "gate_exceptions": "Recorded via record_gate_exception and appended to .github/ai-learning/learning-log.jsonl.",
    "escalation": "Three consecutive gate failures trigger automatic escalation to 00-helping via 00.cross-tier-helper.",
    "cross_tier_helper": "Routes to 00-helping, resolves blocker, returns resolution summary, logs as learning event."
  },
  "certainty_thresholds": {
    "response_suffix": "(Certainty: NN%)",
    "investigate_below_90": "If certainty <90%, stop and investigate before proceeding.",
    "investigate_below_95": "If certainty <95%, ask follow-up questions until requirements and environment are clear."
  },
  "tier_graph": {
    "tiers": [
      {"tier": 0, "label": "Default / Main", "examples": ["Default VS Code Copilot agent"], "user_invocable": false},
      {"tier": 1, "label": "Numbered SDLC Orchestrators", "examples": ["00-helping", "01-planning", "02-researching", "03-red-testing", "04-implementing", "05-green-testing", "06-documenting", "07-logging"], "user_invocable": true},
      {"tier": 2, "label": "Named coordinators / sub-orchestrators", "examples": ["planning-context-coordinator", "solid-split"], "user_invocable": false},
      {"tier": 3, "label": "Hidden scouts and specialists", "examples": ["Boundary Mapper", "Coverage Scout", "Plan Scout"], "user_invocable": false},
      {"tier": 4, "label": "Auxiliaries and one-shot helpers", "examples": ["acceptance-criteria-writer", "file-change-summarizer"], "user_invocable": false}
    ],
    "delegation_rules": [
      "Tier 1 may delegate to Tier 2, 3, or 4.",
      "Tier 2 may delegate to Tier 3 or 4.",
      "Tier 3 may delegate to Tier 4 only.",
      "Tier 4 may not delegate to any agent.",
      "No tier may call a higher-numbered tier except via 00.cross-tier-helper.",
      "user-invocable: true is valid only for Tier 1 agents."
    ],
    "validated_counts": {"Tier 1": 8, "Tier 2": 10, "Tier 3": 35, "Tier 4": 4}
  },
  "skill_and_companion_routing": {
    "skills": "Own durable knowledge: workflow, standards, guardrails, tone models, source-mapping rules, validation expectations, handoff contracts.",
    "companion_agents": "Thin, task-shaped; gather evidence, map boundaries, scout drift, execute one workflow step, defer durable policy to skills.",
    "overlap_rule": "When skill and companion agent overlap, update agent to follow skill.",
    "user_invocable": "Only the eight numbered SDLC orchestrators are directly user-invocable.",
    "catalog_reference": "See .github/agent-skill-routing-table.md for full catalog."
  },
  "canonical_routing_table": {
    "location": ".github/agent-skill-routing-table.md",
    "refresh_command": "npm run agents:routing-table",
    "validate_freshness": "npm run agents:routing-table:gate or node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json",
    "skills_field": "Every .github/agents/*.agent.md file must declare a skills: [...] frontmatter field."
  },
  "workflow_protocols": {
    "mcp_workflow_snapshot": [
      "If get_active_workflow_snapshot returns scope: 'no-active-phase', do not treat as blocking error.",
      "Fallback to direct plan file read for phase/step activation.",
      "Auto-advance Phase 1 Step 01 to [WIP] if plan is brand-new.",
      "When switching sessions, run redirect after Phase 1 Step 01 is [WIP].",
      "Confirm plan and active step alignment before relying on MCP tools.",
      "plans/mcp-active-binding.plans.md is perpetual fallback.",
      "neataptic-workflow-mcp degrades gracefully; neataptic-validation-mcp requires strict [WIP] step."
    ],
    "long_task_logging": [
      "Use compressed logging for long tasks: short entries for changes, remaining work, next target.",
      "Chat communication: brief confirmations and step transitions only.",
      "Prefer tracker files for multi-step tasks.",
      "plans/ is active tracker; plans/completed/ is archive.",
      "Use .plans.md for WIP, decisions, handoff; .logs.md for completed work.",
      "Compress completed .plans.md into closed tracker, update .logs.md, move to plans/completed/.",
      "Strict handoff prompts for active trackers and blocker recovery."
    ],
    "tdd_first_policy": [
      "For behavior changes/regressions/refactors, prefer TDD sequence.",
      "Add/update targeted test to fail first.",
      "Implement code change until test goes green.",
      "Expand coverage after green step.",
      "Keep red/green loop narrow.",
      "Prefer owner-local *.test.ts files, AAA structure, one top-level expect per test."
    ],
    "multi_test_failure_repair": "Invoke test-fix-workflow; do not re-state protocol."
    ,
    "runtime_enforcement": [
      "Strict write/execute actions must prepare the repo-owned runtime proof carrier before tool execution.",
      "Use node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare ... to declare flow ID, delegator chain, required skills, required specialists, plan path, and action class.",
      "PreToolUse and PostToolUse enforce that carrier and log runtime action evidence to .github/ai-learning/learning-log.jsonl.",
      "See .github/runtime-enforcement-contract.md for the canonical runtime enforcement contract and current boundaries."
    ]
  },
  "code_standards": {
    "es2023_policy": [
      "Prefer idiomatic ES2023 syntax for readability and safety.",
      "Use immutable array methods, modern constructs, structuredClone, Error with {cause}, numeric separators, ES modules.",
      "Avoid legacy patterns: in-place sort/reverse/splice, Object.assign for cloning, JSON.parse(JSON.stringify()), index math, CommonJS require."
    ],
    "module_architecture": "Folder-based layout for medium/large modules; orchestration in module.ts, helpers/types/errors/services/constants in separate files.",
    "strict_rules": [
      "Avoid short local identifiers except in tiny idiomatic loops.",
      "Exported classes/functions/constants must have JSDoc with @param/@returns and @example.",
      "Single-expect rule for tests.",
      "Replace magic numbers with named constants and JSDoc.",
      "Step-level inline comments for methods.",
      "Prefer single table/enum for fixed mappings.",
      "Avoid any/unknown types; use precise types or justify exceptions.",
      "Local helper structure: order as locals → calls → return → helpers at end.",
      "Declarative collect → transform → fold flow; isolate type casts.",
      "Multi-pass decomposition: stabilize seams, extract helpers, typed context, orchestration top level."
    ],
    "validation_checklist": [
      "Run npm run build or npx tsc --noEmit -p tsconfig.json.",
      "Run npm run quality:folder for touched folders.",
      "Run npm ci for manifest/tooling changes.",
      "Flag test files with multiple top-level expect per it().",
      "Ensure JSDoc for new exported symbols.",
      "Validate docs/CI for Linux/Chromium.",
      "List flagged legacy patterns and intended replacements."
    ]
  },
  "documentation_standards": {
    "educational_docs": [
      "JSDoc comments compiled into user-facing documentation.",
      "Prefer explanatory, example-driven, conceptual, atemporal docs.",
      "Do not reference internal plans, tracker steps, roadmap phases, pass labels, or chat-only context unless requested.",
      "Keep public docs focused on current concepts, boundaries, invariants, tradeoffs, and reading paths.",
      "Use Mermaid Markdown for diagrams; match neon-retro-arcade style.",
      "Keep examples short, dependency-light, consistent with public API."
    ],
    "generated_readme_handling": [
      "Treat src/**/README.md as read-only; improve JSDoc in source files.",
      "Run npm run docs to refresh generated documentation.",
      "Do not hand-edit generated README; synchronize via docs workflow."
    ],
    "generated_example_publication": [
      "docs/examples/**/index.html is generated; edit source under examples/**/index.html.",
      "Run npm run docs to republish.",
      "Verify generated page after docs run."
    ],
    "ci_sensitive_docs_tooling": [
      "Do not treat local Windows success as sufficient for GitHub Linux runners.",
      "Run npm ci after manifest/lockfile edits.",
      "Run npm run docs for docs/tooling changes.",
      "Account for Chromium sandbox restrictions in Linux CI."
    ],
    "folder_readme_recon": [
      "Read nearest folder README.md before deep code search.",
      "If README is stale/incomplete, improve underlying JSDoc.",
      "Invoke educational-docs for substantial educational improvements."
    ]
  },
  "cross_cutting_policies": {
    "plan_aware_execution": [
      "Invoke plan-alignment for architecture, roadmap, major refactors, export formats, new subsystems.",
      "Agent prompts and summaries should note which README and plan document informed the change.",
      "Keep summaries high-level by default; expand only when requested."
    ],
    "demo_first_library_gap": [
      "Treat demos as evidence of library DX gaps.",
      "Prefer fixing library/public API/defaults/runtime semantics.",
      "Use demo-local compensation only when genuinely demo-specific.",
      "Flag temporary demo-local workarounds as technical debt."
    ],
    "low_context_window_mitigation": [
      "Update source plan document with NEXT: item when context is insufficient.",
      "Provide handoff prompt with relevant context and clear question for companion agent investigation."
    ]
  }
}
