/**
 * @module delegate-skill-coverage.gate.test
 * @description Coverage tests for delegate-skill-coverage.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs',
);

let mockFrontmatterData;
let mockWorkspaceFiles;

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({
    json: argv.includes('--json'),
    help: argv.includes('--help') || argv.includes('-h'),
  }),
  parseFrontmatter: (_text, relPath) => ({
    data: mockFrontmatterData[relPath] ?? {},
    body: '',
    raw: '',
    issues: [],
  }),
  readWorkspaceFile: async (relPath) => mockWorkspaceFiles[relPath] ?? '',
  listMarkdownFiles: async (_dir, filter) => {
    const all = Object.keys(mockWorkspaceFiles);
    return filter ? all.filter(filter) : all;
  },
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGateMain(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./delegate-skill-coverage.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

function desc(name, relPath, data) {
  return { name, relativePath: relPath, contents: '' };
}

describe('delegate-skill-coverage gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockFrontmatterData = {};
    mockWorkspaceFiles = {};
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runDelegateSkillCoverageGate', () => {
    it('returns pass=true when all tier-1/2 agents have execute skill', async () => {
      const rp1 = 'a.agent.md';
      const rp2 = 'b.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['execute', 'other'] },
        [rp2]: { tier: '2', skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [
          desc('a', rp1, ''),
          desc('b', rp2, ''),
        ],
      });
      assert.equal(result.pass, true);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, 'delegate-skill-workflow');
      assert.equal(result.evidence.agentReports.length, 2);
    });

    it('returns pass=false when a tier-1 agent is missing execute skill', async () => {
      const rp1 = 'a.agent.md';
      const rp2 = 'b.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['other'] },
        [rp2]: { tier: 2, skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [
          desc('a', rp1, ''),
          desc('b', rp2, ''),
        ],
      });
      assert.equal(result.pass, false);
      assert.deepEqual(result.evidence.missingDelegateAgents, ['a']);
      assert.ok(result.fixHint.includes('execute'));
      assert.ok(result.fixHint.includes('a'));
    });

    it('skips agents whose tier is not in REQUIRED_TIERS (resolveTier number)', async () => {
      const rp1 = 'a.agent.md';
      const rp2 = 'b.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: 3, skills: ['execute'] },
        [rp2]: { tier: 1, skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [
          desc('a', rp1, ''),
          desc('b', rp2, ''),
        ],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.agentReports.length, 1);
    });

    it('covers resolveTier non-finite string (NaN) and skips', async () => {
      const rp1 = 'a.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: 'abc', skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [desc('a', rp1, '')],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.agentReports.length, 0);
    });

    it('covers resolveTier with non-number non-string (NaN)', async () => {
      const rp1 = 'a.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: true, skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [desc('a', rp1, '')],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.agentReports.length, 0);
    });

    it('covers resolveTier with missing tier (undefined -> NaN)', async () => {
      const rp1 = 'a.agent.md';
      mockFrontmatterData = {
        [rp1]: { skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [desc('a', rp1, '')],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.agentReports.length, 0);
    });

    it('handles non-array skills via toArray (scalar)', async () => {
      const rp1 = 'a.agent.md';
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: 'execute' },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate({
        inventoryLoader: async () => [desc('a', rp1, '')],
      });
      assert.equal(result.pass, false);
      const rep = result.evidence.agentReports[0];
      assert.deepEqual(rep.skills, []);
      assert.equal(rep.hasDelegate, false);
    });

    it('uses default inventoryLoader reading workspace files when not provided', async () => {
      const rp1 = '.github/agents/01-planning.agent.md';
      const rp2 = '.github/agents/04-implementing.agent.md';
      mockWorkspaceFiles = { [rp1]: '', [rp2]: '' };
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['execute'] },
        [rp2]: { tier: 2, skills: ['execute'] },
      };
      const { runDelegateSkillCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./delegate-skill-coverage.gate.mjs'),
      );
      const result = await runDelegateSkillCoverageGate();
      assert.equal(result.pass, true);
    });
  });

  describe('main via import.meta.url guard', () => {
    it('emits JSON with --json when pass=true', async () => {
      const rp1 = '.github/agents/01-planning.agent.md';
      const rp2 = '.github/agents/04-implementing.agent.md';
      mockWorkspaceFiles = { [rp1]: '', [rp2]: '' };
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['execute'] },
        [rp2]: { tier: 2, skills: ['execute'] },
      };
      const logs = await importGateMain(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      const rp1 = '.github/agents/01-planning.agent.md';
      mockWorkspaceFiles = { [rp1]: '' };
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['execute'] },
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('emits JSON with --json when pass=false', async () => {
      const rp1 = '.github/agents/01-planning.agent.md';
      mockWorkspaceFiles = { [rp1]: '' };
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['other'] },
      };
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('emits FAIL text with fixHint without --json when pass=false', async () => {
      const rp1 = '.github/agents/01-planning.agent.md';
      mockWorkspaceFiles = { [rp1]: '' };
      mockFrontmatterData = {
        [rp1]: { tier: 1, skills: ['other'] },
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint')));
      assert.equal(process.exitCode, 1);
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./delegate-skill-coverage.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});