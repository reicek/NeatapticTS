import path from 'node:path';
import { writeFile, unlink } from 'node:fs/promises';

const REPO_ROOT = path.resolve(process.cwd());
const SERVER_PATH = new URL(
  'file:///' + path.resolve(REPO_ROOT, 'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs').replace(/\\/g, '/'),
);
const UTILS_PATH = new URL(
  'file:///' + path.resolve(REPO_ROOT, 'scripts/agent-customization/mcp/mcp-utils.mjs').replace(/\\/g, '/'),
);
const planFile = path.resolve(REPO_ROOT, 'plans', '__debug-active-step.plans.md');
const content = `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
status: '[WIP]'
\`\`\`

## Validation gates

- none
`;
await writeFile(planFile, content.trim(), 'utf8');
const mod = await import(SERVER_PATH);
const utils = await import(UTILS_PATH);
const server = utils.createMcpServer({
  serverName: 'test',
  serverVersion: '0.1.0',
  tools: mod.createWorkflowTools({ planPath: planFile }),
});
const report = await mod.runWorkflowSelfCheck({ server, planPath: planFile });
console.log(JSON.stringify(report, null, 2));
await unlink(planFile);
