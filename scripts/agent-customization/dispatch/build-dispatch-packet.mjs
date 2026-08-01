/**
 * @module build-dispatch-packet
 * @description Pure dispatch-packet builder and delegation policy for the
 *   NeatapticTS dispatch MCP server. Single source of truth for the
 *   complexity classifier, tiered prompt-length limits, and the delegation
 *   validation that the MCP server's `build_dispatch_packet` tool delegates to.
 *
 * Exported so the dispatch MCP server (neataptic-dispatch-mcp.mjs) and unit
 * tests share one implementation, avoiding dual-path logic.
 */

/**
 * Maximum allowed prompt length (characters) for moderate and complex slices.
 * Prompts exceeding this limit are rejected to enforce RAG-based dispatch
 * (short prompt + slice ID, not inline design context). See execute skill
 * Section 2.2 (RAG-Based Dispatch Policy).
 */
export const PROMPT_LENGTH_MAX = 500;

/**
 * Maximum allowed prompt length (characters) for trivial slices. Trivial
 * slices are tightly scoped (one-line fixes, config changes, bundle rebuilds,
 * comments, formatting) and get a stricter prompt budget to reinforce
 * RAG-based dispatch.
 */
export const PROMPT_LENGTH_MAX_TRIVIAL = 200;

/**
 * Complexity levels recognized by the dispatch packet builder. The
 * complexity hint controls the prompt-length budget and the fix-packet
 * fast-path eligibility (see execute skill "Reuse Idle Agent").
 */
export const COMPLEXITY_LEVELS = ['trivial', 'moderate', 'complex'];

/**
 * Default complexity when a caller omits the `complexity` field. Backward
 * compatibility: existing callers that omit complexity are treated as
 * moderate and get the 500-character prompt budget.
 */
export const DEFAULT_COMPLEXITY = 'moderate';

/**
 * Caller tiers allowed to request a dispatch packet.
 */
export const ALLOWED_CALLER_TIERS = [0, 1, 2, 3, 4];

/**
 * Human-readable labels for each tier.
 */
export const TIER_LABELS = {
  1: 'User-invocable phase orchestrators',
  2: 'Tier 2 coordinators',
  3: 'Hidden scouts and specialists',
  4: 'Cross-tier helpers and execution specialists',
};

/**
 * Allowed delegation edges (from tier -> to tier). Delegation must be
 * downward: the target tier must be greater than the caller tier.
 */
export const ALLOWED_EDGES = [
  { from: 0, to: 1 },
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 1, to: 4 },
  { from: 2, to: 3 },
  { from: 2, to: 4 },
  { from: 3, to: 4 },
];

/**
 * Normalize an optional complexity value to one of COMPLEXITY_LEVELS.
 * Unknown / non-string / untrimmed values fall back to DEFAULT_COMPLEXITY
 * ('moderate'). Case-insensitive. This never throws so callers can pass the
 * raw MCP argument through directly.
 *
 * @param {unknown} complexity - Raw complexity hint from the caller.
 * @returns {string} One of COMPLEXITY_LEVELS.
 */
export function normalizeComplexity(complexity) {
  if (typeof complexity !== 'string') return DEFAULT_COMPLEXITY;
  const value = complexity.trim().toLowerCase();
  return COMPLEXITY_LEVELS.includes(value) ? value : DEFAULT_COMPLEXITY;
}

/**
 * Resolve the prompt-length limit for a complexity level. Trivial slices get
 * the stricter 200-character budget; moderate and complex slices get 500.
 *
 * @param {string} complexity - Normalized complexity (trivial|moderate|complex).
 * @returns {number} The maximum allowed prompt length in characters.
 */
export function resolvePromptLengthMax(complexity) {
  return complexity === 'trivial'
    ? PROMPT_LENGTH_MAX_TRIVIAL
    : PROMPT_LENGTH_MAX;
}

/**
 * Validate a delegation request and, if allowed, return a structured dispatch
 * packet. Pure: performs no I/O; the caller supplies the agent inventory.
 *
 * Validation order mirrors the MCP server's historical behavior:
 *   1. Normalize complexity and resolve the prompt-length limit.
 *   2. Reject prompts that exceed the resolved limit (RAG-based dispatch).
 *   3. Find the target agent in the supplied inventory.
 *   4. Validate the caller tier is an allowed integer.
 *   5. Enforce downward delegation (target tier > caller tier).
 *   6. Enforce the user-invocable rule (only Tier 1 may be userInvocable).
 *
 * @param {object} input - The dispatch request.
 * @param {string} input.target_agent - Agent name to dispatch to.
 * @param {number} input.caller_tier - Tier of the caller (0-4).
 * @param {string} [input.prompt=''] - Prompt text embedded verbatim.
 * @param {string} [input.context_tier='default'] - Context tier (default|long_context).
 * @param {string} [input.complexity] - Complexity hint (trivial|moderate|complex). Defaults to moderate.
 * @param {Array<object>} agents - Agent inventory from runCustomizationInventory().
 * @returns {object} Result packet with ok/dispatch_allowed/reason/complexity and,
 *   on success, agent + dispatch_packet. On prompt-length rejection, also
 *   includes prompt_length and prompt_length_max.
 */
export function buildDispatchPacket(
  {
    target_agent,
    caller_tier,
    prompt = '',
    context_tier = 'default',
    complexity,
  },
  agents,
) {
  const complexityNormalized = normalizeComplexity(complexity);
  const promptLengthMax = resolvePromptLengthMax(complexityNormalized);
  const promptText = typeof prompt === 'string' ? prompt : '';
  const contextTier =
    context_tier === 'long_context' ? 'long_context' : 'default';

  // Reject prompts that exceed the maximum allowed length. This enforces
  // RAG-based dispatch: the prompt should be a short instruction with a slice
  // ID, not inline design context. The limit is tiered by complexity.
  if (promptText.length > promptLengthMax) {
    return {
      ok: false,
      dispatch_allowed: false,
      reason:
        `Prompt length ${promptText.length} exceeds the maximum allowed length of ${promptLengthMax} characters for complexity '${complexityNormalized}'. ` +
        'Use RAG-based dispatch: state only the slice ID and a one-line instruction to load context via Cortex MCP.',
      prompt_length: promptText.length,
      prompt_length_max: promptLengthMax,
      complexity: complexityNormalized,
    };
  }

  const target = (agents ?? []).find((agent) => agent.name === target_agent);

  if (!target) {
    return {
      ok: false,
      dispatch_allowed: false,
      reason: `Unknown agent '${target_agent}'`,
      complexity: complexityNormalized,
    };
  }

  const targetTier = Number(target.tier);
  const callerTier = Number(caller_tier);

  if (
    !Number.isInteger(callerTier) ||
    !ALLOWED_CALLER_TIERS.includes(callerTier)
  ) {
    return {
      ok: false,
      dispatch_allowed: false,
      reason: 'caller_tier must be 0, 1, 2, 3, or 4',
      complexity: complexityNormalized,
    };
  }

  if (targetTier <= callerTier) {
    return {
      ok: false,
      dispatch_allowed: false,
      reason: `Delegation from Tier ${callerTier} to Tier ${targetTier} is not allowed because delegation must be downward; the target tier must be greater than the caller tier.`,
      complexity: complexityNormalized,
    };
  }

  if (target.userInvocable === true && targetTier !== 1) {
    return {
      ok: false,
      dispatch_allowed: false,
      reason: `userInvocable is only valid for Tier 1 agents; '${target_agent}' is Tier ${targetTier}`,
      complexity: complexityNormalized,
    };
  }

  return {
    ok: true,
    dispatch_allowed: true,
    reason: `Tier ${callerTier} may delegate to Tier ${targetTier}`,
    complexity: complexityNormalized,
    agent: {
      name: target.name,
      tier: targetTier,
      tier_label: TIER_LABELS[targetTier] ?? `Tier ${targetTier}`,
      model: target.model ?? null,
      skills: target.skills,
      agents: target.agents,
      tools: target.tools,
      userInvocable: target.userInvocable,
      file: target.path,
    },
    dispatch_packet: {
      agent_type: target.name,
      name: target.name,
      description: target.description,
      model: target.model ?? null,
      prompt: promptText,
      context_tier: contextTier,
      complexity: complexityNormalized,
      skills: target.skills,
    },
  };
}
