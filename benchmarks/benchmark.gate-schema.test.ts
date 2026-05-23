/**
 * Red-phase contract tests for Tier-1 gate scripts and pilot flow YAML resolution.
 *
 * These tests are intentionally red until Phase 2 Step 04 implements:
 * - `scripts/agent-customization/gates/{plan-sync,step-packet,agent-graph,learning-event}.gate.mjs`
 * - `.github/flows/{04.scoped-fix,04.refactor,04.coverage-repair}.flow.yml`
 *
 * Gate contract (from Agentic_Flows_and_Gates_Upgrade.plans.md Step 01):
 *   pass: boolean, evidence: object, fixHint: string, owner: string
 */
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const ROOT = path.resolve(__dirname, '..');
const GATES_DIR = path.join(ROOT, 'scripts', 'agent-customization', 'gates');
const FLOWS_DIR = path.join(ROOT, '.github', 'flows');

/**
 * Run a gate script with --json flag and return parsed stdout.
 * Throws if the script exits non-zero or stdout is not valid JSON.
 */
function runGateJson(scriptName: string): Record<string, unknown> {
  const scriptPath = path.join(GATES_DIR, scriptName);
  const result = spawnSync('node', [scriptPath, '--json'], {
    encoding: 'utf8',
    timeout: 10_000,
  });
  // JSON.parse throws if stdout is empty or invalid — test goes red
  return JSON.parse(result.stdout) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Tier-1 gate output contract
// ---------------------------------------------------------------------------

describe('Tier-1 gate scripts — output contract', () => {
  // plan-sync gate
  describe('plan-sync gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('plan-sync.gate.mjs')['pass']).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('plan-sync.gate.mjs');
      expect(output['evidence'] !== null && typeof output['evidence'] === 'object').toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(typeof runGateJson('plan-sync.gate.mjs')['fixHint']).toBe('string');
    });

    it('returns an owner string', () => {
      expect(typeof runGateJson('plan-sync.gate.mjs')['owner']).toBe('string');
    });
  });

  // step-packet gate
  describe('step-packet gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('step-packet.gate.mjs')['pass']).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('step-packet.gate.mjs');
      expect(output['evidence'] !== null && typeof output['evidence'] === 'object').toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(typeof runGateJson('step-packet.gate.mjs')['fixHint']).toBe('string');
    });

    it('returns an owner string', () => {
      expect(typeof runGateJson('step-packet.gate.mjs')['owner']).toBe('string');
    });
  });

  // agent-graph gate
  describe('agent-graph gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('agent-graph.gate.mjs')['pass']).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('agent-graph.gate.mjs');
      expect(output['evidence'] !== null && typeof output['evidence'] === 'object').toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(typeof runGateJson('agent-graph.gate.mjs')['fixHint']).toBe('string');
    });

    it('returns an owner string', () => {
      expect(typeof runGateJson('agent-graph.gate.mjs')['owner']).toBe('string');
    });
  });

  // learning-event gate
  describe('learning-event gate', () => {
    it('returns a pass boolean', () => {
      expect(typeof runGateJson('learning-event.gate.mjs')['pass']).toBe('boolean');
    });

    it('returns an evidence object', () => {
      const output = runGateJson('learning-event.gate.mjs');
      expect(output['evidence'] !== null && typeof output['evidence'] === 'object').toBe(true);
    });

    it('returns a fixHint string', () => {
      expect(typeof runGateJson('learning-event.gate.mjs')['fixHint']).toBe('string');
    });

    it('returns an owner string', () => {
      expect(typeof runGateJson('learning-event.gate.mjs')['owner']).toBe('string');
    });
  });
});

// ---------------------------------------------------------------------------
// Pilot flow YAML resolution
// ---------------------------------------------------------------------------

describe('Pilot flow YAML resolution', () => {
  const PILOT_FLOWS = [
    '04.scoped-fix.flow.yml',
    '04.refactor.flow.yml',
    '04.coverage-repair.flow.yml',
  ] as const;

  for (const flowFile of PILOT_FLOWS) {
    describe(flowFile, () => {
      it('exists in the .github/flows directory', () => {
        expect(fs.existsSync(path.join(FLOWS_DIR, flowFile))).toBe(true);
      });
    });
  }
});
