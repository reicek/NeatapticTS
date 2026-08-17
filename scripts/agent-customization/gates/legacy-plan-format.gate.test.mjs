/**
 * @module legacy-plan-format.gate.test
 * @description Coverage tests for legacy-plan-format.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/legacy-plan-format.gate.mjs',
);

// --- Mock state -----------------------------------------------------------
let mockReadFileFn;
let mockReaddirFn;
let mockStatFn;
let mockParseYamlFn;
let mockParseYamlCallCount;

jest.unstable_mockModule('node:fs/promises', () => ({
  readdir: async (dirPath) => mockReaddirFn(dirPath),
  readFile: async (filePath, opts) => mockReadFileFn(filePath, opts),
  stat: async (filePath) => mockStatFn(filePath),
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({
    json: argv.includes('--json'),
    help: argv.includes('--help'),
  }),
  repoRoot: REPO_ROOT,
  parsePlanYamlBlock: (rawBlock) => {
    mockParseYamlCallCount++;
    return mockParseYamlFn(rawBlock, mockParseYamlCallCount);
  },
}));

// --- Helper ---------------------------------------------------------------
async function importGate(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    const originalArgv = process.argv;
    process.argv = [process.execPath, GATE_PATH, ...argv];
    await import('./legacy-plan-format.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
    process.argv = originalArgv;
  } finally {
    console.log = originalLog;
  }
  return logs;
}

// Content helpers — build plan text with YAML blocks and headings
function yamlBlock(content) {
  return '```yaml\n' + content + '\n```';
}

function planWithHeading(heading, yamlContent) {
  return heading + '\n\n' + yamlBlock(yamlContent);
}

// --- Tests ----------------------------------------------------------------
describe('legacy-plan-format gate', () => {
  let originalExitCode;

  beforeEach(() => {
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
    jest.resetModules();
    mockReadFileFn = async () => '';
    mockReaddirFn = async () => [];
    mockStatFn = async () => ({ isFile: () => true });
    mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
    mockParseYamlCallCount = 0;
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('collectPlanFiles', () => {
    it('returns pass=true when no plan directories are readable', async () => {
      mockReaddirFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.plansScanned, 0);
      assert.equal(parsed.evidence.blocksChecked, 0);
    });

    it('skips hidden files starting with _ or .', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) {
          return ['_hidden.plans.md', '.hidden.plans.md', 'plan1.plans.md'];
        }
        return [];
      };
      mockReadFileFn = async () => 'no yaml blocks here';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      // Only plan1.plans.md should be scanned (not _hidden or .hidden)
      assert.equal(parsed.evidence.plansScanned, 1);
    });

    it('skips non-.plans.md files', async () => {
      mockReaddirFn = async () => {
        // First call: plans/, second call: plans/completed/
        return ['plan1.plans.md', 'readme.md', 'data.txt'];
      };
      mockReadFileFn = async () => 'no yaml blocks';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      // Only .plans.md files counted, but both dirs return same entries
      // plans/ has plan1.plans.md, plans/completed/ has plan1.plans.md
      assert.equal(parsed.evidence.plansScanned, 2);
    });

    it('skips entries where stat shows not a file', async () => {
      mockReaddirFn = async () => ['plan1.plans.md'];
      mockStatFn = async () => ({ isFile: () => false });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.plansScanned, 0);
    });

    it('scans both plans/ and plans/completed/ directories', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['active.plans.md'];
        return ['completed1.plans.md'];
      };
      mockReadFileFn = async () => 'no yaml here';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.plansScanned, 2);
    });
  });

  describe('runLegacyFormatGate — block skipping', () => {
    it('skips plan files that cannot be read', async () => {
      mockReaddirFn = async () => {
        // Only first dir has files
        return ['plan1.plans.md'];
      };
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 0);
    });

    it('returns pass=true when plan has no YAML blocks', async () => {
      mockReaddirFn = async () => {
        return ['plan1.plans.md'];
      };
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () => 'No YAML blocks in this file.\n';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 0);
    });

    it('skips blocks with DONE heading status', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [DONE]',
          "status: '[WIP]'\nphase: 1",
        );
      // parsePlanYamlBlock should NOT be called because headingStatus === 'DONE'
      mockParseYamlFn = () => {
        throw new Error('should not be called');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });

    it('skips blocks with DONE yaml status', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[DONE]'\nphase: 1",
        );
      mockParseYamlFn = () => {
        throw new Error('should not be called');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });

    it('skips PlanUpdate blocks', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "PlanUpdate: something\nstatus: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => {
        throw new Error('should not be called');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });

    it('records YAML parse errors as legacy blocks', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => {
        throw new Error('YAML parse error');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.legacyBlocks.length, 1);
      assert.ok(parsed.evidence.legacyBlocks[0].reason.includes('YAML parse error'));
    });

    it('skips blocks where metadata.phase is undefined', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'",
        );
      mockParseYamlFn = () => ({ expansion: 'none' }); // no phase property
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });
  });

  describe('runLegacyFormatGate — legacy block detection', () => {
    it('detects legacy block missing expansion field', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1 }); // no expansion, no agent
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.legacyBlocks.length, 1);
      assert.ok(
        parsed.evidence.legacyBlocks[0].reasons.includes(
          'missing expansion field',
        ),
      );
    });

    it('detects legacy block with deprecated agent field', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none', agent: 'foo' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.legacyBlocks.length, 1);
      assert.ok(
        parsed.evidence.legacyBlocks[0].reasons.includes(
          'deprecated agent field',
        ),
      );
    });

    it('detects legacy block with deprecated agent_file field', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({
        phase: 1,
        expansion: 'none',
        agent_file: 'foo.mjs',
      });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.legacyBlocks.length, 1);
      assert.ok(
        parsed.evidence.legacyBlocks[0].reasons.includes(
          'deprecated agent_file field',
        ),
      );
    });

    it('detects legacy block with all three reasons', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({
        phase: 1,
        agent: 'foo',
        agent_file: 'bar.mjs',
      });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.legacyBlocks[0].reasons.length, 3);
    });

    it('does not flag block with valid modern format (has expansion, no agent)', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });
  });

  describe('extractStatusFromYaml and findNearestHeadingStatus coverage', () => {
    it('handles PLANNED status and step heading', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '#### Step 1.1: Do something [WIP]',
          "status: '[PLANNED]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
    });

    it('handles block with no heading (findNearestHeadingStatus returns null)', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      // YAML block at start of file — no heading before it
      mockReadFileFn = async () =>
        yamlBlock("status: '[WIP]'\nphase: 1");
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
    });

    it('handles block with non-heading lines before it', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        'Some text.\nMore text.\n\n' +
        yamlBlock("status: '[WIP]'\nphase: 1");
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
    });

    it('handles block with no status line (extractStatusFromYaml returns null)', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          'phase: 1',
        );
      mockParseYamlFn = () => ({ phase: 1, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 1);
    });

    it('handles multiple YAML blocks in one file', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        '### Phase 1: Setup [DONE]\n\n' +
        yamlBlock("status: '[WIP]'\nphase: 1") +
        '\n\n### Phase 2: Build [WIP]\n\n' +
        yamlBlock("status: '[WIP]'\nphase: 2");
      // First block skipped (DONE heading), second block is modern
      mockParseYamlFn = () => ({ phase: 2, expansion: 'none' });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.blocksChecked, 2);
      assert.equal(parsed.evidence.legacyBlocks.length, 0);
    });
  });

  describe('output format', () => {
    it('emits JSON with --json when pass=true', async () => {
      mockReaddirFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text without --json when pass=true', async () => {
      mockReaddirFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('legacy-plan-format gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with fixHint when pass=false', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1 });
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      let readdirCalls = 0;
      mockReaddirFn = async () => {
        readdirCalls++;
        if (readdirCalls === 1) return ['plan1.plans.md'];
        return [];
      };
      mockReadFileFn = async () =>
        planWithHeading(
          '### Phase 1: Setup [WIP]',
          "status: '[WIP]'\nphase: 1",
        );
      mockParseYamlFn = () => ({ phase: 1 });
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });
  });
});