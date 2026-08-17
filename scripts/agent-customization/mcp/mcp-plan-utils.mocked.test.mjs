/**
 * @fileoverview Coverage test for mcp-plan-utils.mjs using jest.unstable_mockModule
 * to force parsePlanYamlBlock to throw — covers the parsePhaseMetadata catch block (line 389).
 */

import { jest } from '@jest/globals';
import { writeFile, unlink } from 'node:fs/promises';
import path from 'node:path';

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parsePlanYamlBlock: jest.fn((yamlText) => {
    if (typeof yamlText === 'string' && yamlText.includes('FORCE_THROW')) {
      throw new Error('forced parse error');
    }
    return { agent: 'test-agent', goal: 'test-goal', validation: ['npm test'] };
  }),
  issue: jest.fn(),
  summarizeIssues: jest.fn(() => ({
    name: 'mock',
    errorCount: 0,
    warningCount: 0,
    issues: [],
  })),
  writeReport: jest.fn(),
}));

let loadActivePlanContext;
let MCP_REPO_ROOT;
let PLANS_DIR;

beforeAll(async () => {
  const mcpUtils = await import('./mcp-utils.mjs');
  MCP_REPO_ROOT = mcpUtils.MCP_REPO_ROOT;
  PLANS_DIR = path.join(MCP_REPO_ROOT, 'plans');
  const mcpPlanUtils = await import('./mcp-plan-utils.mjs');
  loadActivePlanContext = mcpPlanUtils.loadActivePlanContext;
});

const createdFiles = [];

afterEach(async () => {
  for (const filePath of createdFiles) {
    try {
      await unlink(filePath);
    } catch {
      // already gone
    }
  }
  createdFiles.length = 0;
});

const PHASE_THROW_PLAN = `# Phase Throw Test Plan

## Implementation phases

### Phase 1 — Throw Phase [WIP]

\`\`\`yaml
FORCE_THROW: true
expansion: steps
auto_expand: false
\`\`\`

#### Step 01 — Normal Step [WIP]

\`\`\`yaml
agent: 'test-agent'
goal: 'test goal'
validation:
  - 'npm test'
\`\`\`

**Step objective:** Test the catch block.

**Required validation:**
- \`npm test\`

## Validation gates
`;

describe('parsePhaseMetadata catch block (mocked parsePlanYamlBlock)', () => {
  it('returns empty object when parsePlanYamlBlock throws', async () => {
    const filePath = path.join(PLANS_DIR, 'test-mock-phase-throw.plans.md');
    await writeFile(filePath, PHASE_THROW_PLAN, 'utf8');
    createdFiles.push(filePath);

    const planPath = 'plans/test-mock-phase-throw.plans.md';
    const ctx = await loadActivePlanContext(planPath);

    // Phase metadata should be {} from the catch block
    expect(ctx.phaseMetadata).toEqual({});
    // Step metadata should be from the mock (no FORCE_THROW in step YAML)
    expect(ctx.activeStep.metadata.agent).toBe('test-agent');
  });
});