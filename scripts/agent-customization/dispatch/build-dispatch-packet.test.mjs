/**
 * @module build-dispatch-packet.test
 * @description Jest unit tests for the pure dispatch-packet builder.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly. Covers the complexity classifier, the tiered
 * 200/500 prompt-length limits, backward compatibility for callers that omit
 * complexity, and the core delegation validation rules.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

import {
  ALLOWED_EDGES,
  buildDispatchPacket,
  COMPLEXITY_LEVELS,
  DEFAULT_COMPLEXITY,
  normalizeComplexity,
  PROMPT_LENGTH_MAX,
  PROMPT_LENGTH_MAX_TRIVIAL,
  resolvePromptLengthMax,
} from './build-dispatch-packet.mjs';

/**
 * A small, deterministic agent inventory used by every test. Mirrors the
 * shape returned by runCustomizationInventory() so the pure builder exercises
 * the same field accesses as the live MCP server.
 */
const AGENTS = [
  {
    name: '01-planning',
    tier: 1,
    model: 'glm-5.2',
    skills: ['planning'],
    agents: [],
    tools: ['view'],
    userInvocable: true,
    path: '.github/agents/01-planning.agent.md',
    description: 'Planning orchestrator.',
  },
  {
    name: 'implementation-executor',
    tier: 3,
    model: 'glm-5.2',
    skills: ['implementation-standards'],
    agents: [],
    tools: ['edit', 'view'],
    userInvocable: false,
    path: '.github/agents/implementation-executor.agent.md',
    description: 'Scoped code editor.',
  },
  {
    name: 'synthetic-userinvocable-tier3',
    tier: 3,
    model: 'glm-5.2',
    skills: [],
    agents: [],
    tools: [],
    userInvocable: true,
    path: '.github/agents/synthetic.agent.md',
    description: 'Synthetic invalid agent.',
  },
  {
    name: 'tier5-no-model-agent',
    tier: 5,
    skills: [],
    agents: [],
    tools: [],
    userInvocable: false,
    path: '.github/agents/tier5-no-model-agent.agent.md',
    description: 'Synthetic tier 5 agent without a model property.',
  },
];

describe('build-dispatch-packet constants', () => {
  it('exposes the tiered prompt-length limits and complexity levels', () => {
    assert.strictEqual(PROMPT_LENGTH_MAX, 500);
    assert.strictEqual(PROMPT_LENGTH_MAX_TRIVIAL, 200);
    assert.deepStrictEqual(COMPLEXITY_LEVELS, [
      'trivial',
      'moderate',
      'complex',
    ]);
    assert.strictEqual(DEFAULT_COMPLEXITY, 'moderate');
  });

  it('exposes the allowed delegation edges unchanged', () => {
    assert.deepStrictEqual(ALLOWED_EDGES, [
      { from: 0, to: 1 },
      { from: 1, to: 2 },
      { from: 1, to: 3 },
      { from: 1, to: 4 },
      { from: 2, to: 3 },
      { from: 2, to: 4 },
      { from: 3, to: 4 },
    ]);
  });
});

describe('normalizeComplexity', () => {
  it('returns the normalized value for valid levels (case-insensitive, trimmed)', () => {
    assert.strictEqual(normalizeComplexity('trivial'), 'trivial');
    assert.strictEqual(normalizeComplexity('  Trivial '), 'trivial');
    assert.strictEqual(normalizeComplexity('MODERATE'), 'moderate');
    assert.strictEqual(normalizeComplexity('complex'), 'complex');
  });

  it('falls back to moderate for unknown or non-string values', () => {
    assert.strictEqual(normalizeComplexity('unknown'), 'moderate');
    assert.strictEqual(normalizeComplexity(undefined), 'moderate');
    assert.strictEqual(normalizeComplexity(null), 'moderate');
    assert.strictEqual(normalizeComplexity(42), 'moderate');
    assert.strictEqual(normalizeComplexity(''), 'moderate');
  });
});

describe('resolvePromptLengthMax', () => {
  it('returns 200 for trivial and 500 for moderate/complex', () => {
    assert.strictEqual(resolvePromptLengthMax('trivial'), 200);
    assert.strictEqual(resolvePromptLengthMax('moderate'), 500);
    assert.strictEqual(resolvePromptLengthMax('complex'), 500);
  });
});

describe('buildDispatchPacket — complexity field', () => {
  it('defaults to moderate when complexity is omitted (backward compatible)', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'Fix the typo.',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.complexity, 'moderate');
    assert.strictEqual(result.dispatch_packet.complexity, 'moderate');
  });

  it('accepts explicit trivial complexity and echoes it in the packet', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'Fix the typo.',
        complexity: 'trivial',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.complexity, 'trivial');
    assert.strictEqual(result.dispatch_packet.complexity, 'trivial');
  });

  it('accepts explicit complex complexity and echoes it in the packet', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'Refactor the GPU pipeline.',
        complexity: 'complex',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.complexity, 'complex');
    assert.strictEqual(result.dispatch_packet.complexity, 'complex');
  });

  it('normalizes an unknown complexity to moderate (backward compatible)', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        complexity: 'banana',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.complexity, 'moderate');
  });
});

describe('buildDispatchPacket — tiered prompt-length limits', () => {
  it('enforces a 200-character limit for trivial slices', () => {
    const accepted = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(200),
        complexity: 'trivial',
      },
      AGENTS,
    );
    assert.strictEqual(accepted.ok, true);
    assert.strictEqual(accepted.complexity, 'trivial');

    const rejected = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(201),
        complexity: 'trivial',
      },
      AGENTS,
    );
    assert.strictEqual(rejected.ok, false);
    assert.strictEqual(rejected.dispatch_allowed, false);
    assert.strictEqual(rejected.prompt_length, 201);
    assert.strictEqual(rejected.prompt_length_max, 200);
    assert.strictEqual(rejected.complexity, 'trivial');
    assert.match(rejected.reason, /200/);
  });

  it('enforces a 500-character limit for moderate slices (default)', () => {
    const accepted = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(500),
      },
      AGENTS,
    );
    assert.strictEqual(accepted.ok, true);
    assert.strictEqual(accepted.complexity, 'moderate');

    const rejected = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(501),
      },
      AGENTS,
    );
    assert.strictEqual(rejected.ok, false);
    assert.strictEqual(rejected.dispatch_allowed, false);
    assert.strictEqual(rejected.prompt_length, 501);
    assert.strictEqual(rejected.prompt_length_max, 500);
    assert.strictEqual(rejected.complexity, 'moderate');
    assert.match(rejected.reason, /500/);
  });

  it('enforces a 500-character limit for complex slices', () => {
    const accepted = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(500),
        complexity: 'complex',
      },
      AGENTS,
    );
    assert.strictEqual(accepted.ok, true);
    assert.strictEqual(accepted.complexity, 'complex');

    const rejected = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'x'.repeat(501),
        complexity: 'complex',
      },
      AGENTS,
    );
    assert.strictEqual(rejected.ok, false);
    assert.strictEqual(rejected.dispatch_allowed, false);
    assert.strictEqual(rejected.prompt_length_max, 500);
    assert.strictEqual(rejected.complexity, 'complex');
  });

  it('still rejects an overlong prompt before looking up the agent', () => {
    // An overlong prompt with an unknown agent still reports the prompt
    // rejection (prompt-length fields present), not the unknown-agent reason.
    const rejected = buildDispatchPacket(
      {
        target_agent: '__nonexistent__',
        caller_tier: 1,
        prompt: 'x'.repeat(501),
      },
      AGENTS,
    );
    assert.strictEqual(rejected.ok, false);
    assert.strictEqual(rejected.dispatch_allowed, false);
    assert.strictEqual(typeof rejected.prompt_length, 'number');
    assert.strictEqual(typeof rejected.prompt_length_max, 'number');
  });
});

describe('buildDispatchPacket — delegation validation', () => {
  it('returns a success packet with the expected shape for a downward delegation', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: 'Implement the slice.',
        context_tier: 'long_context',
        complexity: 'moderate',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.dispatch_allowed, true);
    assert.strictEqual(result.agent.tier, 3);
    assert.strictEqual(
      result.agent.tier_label,
      'Hidden scouts and specialists',
    );
    assert.strictEqual(
      result.dispatch_packet.agent_type,
      'implementation-executor',
    );
    assert.strictEqual(result.dispatch_packet.context_tier, 'long_context');
    assert.strictEqual(result.dispatch_packet.prompt, 'Implement the slice.');
  });

  it('rejects an unknown target agent', () => {
    const result = buildDispatchPacket(
      { target_agent: '__nope__', caller_tier: 1 },
      AGENTS,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(result.reason, /Unknown agent/);
    assert.strictEqual(result.complexity, 'moderate');
  });

  it('rejects an upward delegation', () => {
    const result = buildDispatchPacket(
      { target_agent: '01-planning', caller_tier: 3 },
      AGENTS,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(
      result.reason,
      /target tier must be greater than the caller tier/i,
    );
  });

  it('rejects a userInvocable agent that is not Tier 1', () => {
    const result = buildDispatchPacket(
      { target_agent: 'synthetic-userinvocable-tier3', caller_tier: 1 },
      AGENTS,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(
      result.reason,
      /userInvocable is only valid for Tier 1 agents/i,
    );
  });

  it('rejects an invalid caller tier', () => {
    const result = buildDispatchPacket(
      { target_agent: 'implementation-executor', caller_tier: 7 },
      AGENTS,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(result.reason, /caller_tier must be 0, 1, 2, 3, or 4/);
  });

  it('handles an empty agents inventory gracefully', () => {
    const result = buildDispatchPacket(
      { target_agent: 'implementation-executor', caller_tier: 1 },
      [],
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(result.reason, /Unknown agent/);
  });
});

describe('buildDispatchPacket — branch coverage edge cases', () => {
  it('coerces a non-string prompt to an empty string (typeof prompt !== "string")', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'implementation-executor',
        caller_tier: 1,
        prompt: null,
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.dispatch_packet.prompt, '');
  });

  it('handles a null agents inventory via the ?? [] fallback', () => {
    const result = buildDispatchPacket(
      { target_agent: 'implementation-executor', caller_tier: 1 },
      null,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.dispatch_allowed, false);
    assert.match(result.reason, /Unknown agent/);
  });

  it('uses fallback tier label and null model for an unlisted tier and missing model', () => {
    const result = buildDispatchPacket(
      {
        target_agent: 'tier5-no-model-agent',
        caller_tier: 4,
        prompt: 'Do the thing.',
      },
      AGENTS,
    );
    assert.strictEqual(result.ok, true);
    assert.strictEqual(result.dispatch_allowed, true);
    assert.strictEqual(result.agent.tier, 5);
    assert.strictEqual(result.agent.tier_label, 'Tier 5');
    assert.strictEqual(result.agent.model, null);
    assert.strictEqual(result.dispatch_packet.model, null);
    assert.strictEqual(result.dispatch_packet.prompt, 'Do the thing.');
  });
});
