/**
 * @module devtools-coverage.gate.test
 * @description Coverage tests for devtools-coverage.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/devtools-coverage.gate.mjs',
);

const AGENTS_DIR = '.github/agents';
const ALL_SPECIALISTS = [
  'performance-trace-specialist',
  'browser-ui-specialist',
  'browser-memory-specialist',
];

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
  listMarkdownFiles: async () => Object.keys(mockWorkspaceFiles),
}));

function relPathFor(agentName) {
  return `${AGENTS_DIR}/${agentName}.agent.md`;
}

function fullAgent(skills = ['devtools', 'other'], agents = ALL_SPECIALISTS) {
  return { skills, agents };
}

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
      await import('./devtools-coverage.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('devtools-coverage gate', () => {
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

  describe('runDevtoolsCoverageGate', () => {
    it('returns pass=true when all agents have skill and specialists', async () => {
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate({
        agentLoader: async () => '---\n---',
      });
      assert.equal(result.pass, true);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, 'devtools-workflow');
      assert.equal(result.evidence.agentReports.length, 2);
    });

    it('returns pass=false when an agent is missing the devtools skill', async () => {
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(['other'], ALL_SPECIALISTS),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate({
        agentLoader: async () => '',
      });
      assert.equal(result.pass, false);
      assert.deepEqual(result.evidence.missingSkillAgents, ['03-red-testing']);
      assert.ok(result.fixHint.includes("'devtools'"));
      assert.ok(result.fixHint.includes('03-red-testing'));
    });

    it('returns pass=false when an agent is missing specialists', async () => {
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(
          ['devtools'],
          ['performance-trace-specialist', 'browser-ui-specialist'],
        ),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate({
        agentLoader: async () => '',
      });
      assert.equal(result.pass, false);
      assert.deepEqual(result.evidence.missingSkillAgents, []);
      assert.equal(result.evidence.missingSpecialistAgents.length, 1);
      assert.equal(
        result.evidence.missingSpecialistAgents[0].specialist,
        'browser-memory-specialist',
      );
      assert.ok(result.fixHint.includes('specialist'));
    });

    it('handles non-array skills via toArray (scalar)', async () => {
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: {
          skills: 'devtools',
          agents: ALL_SPECIALISTS,
        },
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate({
        agentLoader: async () => '',
      });
      assert.equal(result.pass, false);
      const red = result.evidence.agentReports.find(
        (r) => r.name === '03-red-testing',
      );
      assert.deepEqual(red.skills, []);
      assert.equal(red.hasSkill, false);
    });

    it('handles missing skills and agents fields via toArray (undefined)', async () => {
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: {},
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate({
        agentLoader: async () => '',
      });
      assert.equal(result.pass, false);
      const red = result.evidence.agentReports.find(
        (r) => r.name === '03-red-testing',
      );
      assert.deepEqual(red.skills, []);
      assert.deepEqual(red.agents, []);
      assert.equal(red.hasSkill, false);
      assert.equal(red.missingSpecialists.length, 3);
    });

    it('uses default agentLoader reading workspace files when not provided', async () => {
      mockWorkspaceFiles = {
        [relPathFor('03-red-testing')]: '---\n---',
        [relPathFor('05-green-testing')]: '---\n---',
      };
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const { runDevtoolsCoverageGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./devtools-coverage.gate.mjs'),
      );
      const result = await runDevtoolsCoverageGate();
      assert.equal(result.pass, true);
    });
  });

  describe('main via import.meta.url guard', () => {
    it('emits JSON with --json when pass=true', async () => {
      mockWorkspaceFiles = {
        [relPathFor('03-red-testing')]: '',
        [relPathFor('05-green-testing')]: '',
      };
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const logs = await importGateMain(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      mockWorkspaceFiles = {
        [relPathFor('03-red-testing')]: '',
        [relPathFor('05-green-testing')]: '',
      };
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockWorkspaceFiles = {
        [relPathFor('03-red-testing')]: '',
        [relPathFor('05-green-testing')]: '',
      };
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(['other'], ALL_SPECIALISTS),
        [relPathFor('05-green-testing')]: fullAgent(),
      };
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('emits FAIL text with fixHint without --json when pass=false', async () => {
      mockWorkspaceFiles = {
        [relPathFor('03-red-testing')]: '',
        [relPathFor('05-green-testing')]: '',
      };
      mockFrontmatterData = {
        [relPathFor('03-red-testing')]: fullAgent(['other'], ALL_SPECIALISTS),
        [relPathFor('05-green-testing')]: fullAgent(),
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
          await import('./devtools-coverage.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});
