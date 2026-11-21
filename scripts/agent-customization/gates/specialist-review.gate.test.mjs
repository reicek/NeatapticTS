/**
 * @module specialist-review.gate.test
 * @description Coverage tests for specialist-review.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/specialist-review.gate.mjs',
);

// --- Mock state -----------------------------------------------------------
let mockReadFileSyncFn;
let mockReaddirSyncFn;
let mockReadFileCalls;

jest.unstable_mockModule('node:fs', () => ({
  readFileSync: (p, opts) => {
    const key = String(p);
    mockReadFileCalls[key] = (mockReadFileCalls[key] || 0) + 1;
    return mockReadFileSyncFn(p, opts, mockReadFileCalls[key]);
  },
  readdirSync: (p) => mockReaddirSyncFn(p),
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => {
    const planArg = argv.find((a) => a.startsWith('--plan='));
    return {
      json: argv.includes('--json'),
      help: argv.includes('--help'),
      // Mirror PRODUCTION `parseArgs`: when `--plan` is absent it injects a
      // DEFAULT plan — a COMPLETED plan under plans/completed/. The gate must
      // NOT trust that default; the mock reproduces it so the tests actually
      // exercise the stale-default hazard.
      plan: planArg
        ? planArg.slice('--plan='.length)
        : 'plans/completed/Agentic_Workflow_Architecture.plans.md',
    };
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
    await import('./specialist-review.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
    process.argv = originalArgv;
  } finally {
    console.log = originalLog;
  }
  return logs;
}

// Content helpers
const WIP_WITH_SPECIALIST =
  '[WIP]\n## VALIDATION_EVIDENCE\nimplementation-pattern-scout: APPROVE\n';
const WIP_WITH_TRIVIAL = '[WIP]\n## VALIDATION_EVIDENCE\nseverity: TRIVIAL\n';
const WIP_NO_EVIDENCE =
  '[WIP]\nSome plan content without validation evidence section.\n';
const WIP_NO_MARKERS =
  '[WIP]\n## VALIDATION_EVIDENCE\nSome evidence but no specialist markers.\n';
// CANONICAL heading form — the shape live plans carry.
const WIP_CANONICAL_HEADING =
  '[WIP]\n## Latest validation evidence\nimplementation-pattern-scout: APPROVE\n';
const WIP_CANONICAL_HEADING_NO_MARKERS =
  '[WIP]\n## Latest validation evidence\nSome evidence but no specialist markers.\n';
const WIP_CANONICAL_HEADING_TRIVIAL =
  '[WIP]\n## Latest validation evidence\nseverity: TRIVIAL\n';
// A canonical heading whose body is empty carries no evidence at all.
const WIP_CANONICAL_HEADING_EMPTY =
  '[WIP]\n## Latest validation evidence\n## Next section\nbody\n';
// A `###` sub-block under the canonical `##` heading stays INSIDE the
// evidence region (the region ends at the next SAME-OR-SHALLOWER heading).
const WIP_CANONICAL_HEADING_SUBBLOCK = [
  '[WIP]',
  '## Latest validation evidence',
  '### Slice 5 evidence',
  'specialist_review: APPROVE',
  '## Next section',
  'unrelated content',
  '',
].join('\n');
// The canonical heading quoted inside a fenced code block is NOT a live
// section and must not be treated as evidence.
const WIP_HEADING_INSIDE_CODE_FENCE = [
  '[WIP]',
  '```md',
  '## Latest validation evidence',
  'specialist_review: APPROVE',
  '```',
  '',
].join('\n');
// A long multi-step plan carries one evidence block per step packet, so the
// reviewed step's block is NOT the first one in the file.
const WIP_TWO_EVIDENCE_SECTIONS = [
  '[WIP]',
  '## Step 14C',
  '```yaml',
  'PlanUpdate:',
  '  slice_id: "fix-packet-closed-step"',
  '  validation_evidence:',
  '    - "npx tsc --noEmit -> pass"',
  '```',
  '## Step 14J',
  '```yaml',
  'PlanUpdate:',
  '  slice_id: "active-step"',
  '  validation_evidence:',
  '    - "specialist_review: APPROVE"',
  '```',
  '',
].join('\n');
// Two canonical sections — the reviewed one is NOT the first.
const WIP_TWO_CANONICAL_SECTIONS = [
  '[WIP]',
  '## Latest validation evidence',
  'step 14C: tsc pass',
  '## Step 14J',
  '## Latest validation evidence',
  'specialist_review: APPROVE',
  '',
].join('\n');
const NO_WIP = 'Completed plan with no WIP markers.\n';
// The stale COMPLETED plan that `parseArgs` injects as its default. It only
// carries the LEGACY token, so a gate that trusts the default reports a
// green run that never touched the active plan.
const STALE_DEFAULT_PLAN =
  '[WIP]\n## VALIDATION_EVIDENCE\nimplementation-pattern-scout: APPROVE\n';

// --- Tests ----------------------------------------------------------------
describe('specialist-review gate', () => {
  let originalExitCode;

  beforeEach(() => {
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
    jest.resetModules();
    mockReadFileCalls = {};
    mockReadFileSyncFn = () => '';
    mockReaddirSyncFn = () => [];
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('no WIP plans', () => {
    it('returns pass=true when no plans found (empty directory)', async () => {
      mockReaddirSyncFn = () => [];
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=true when plans do not contain [WIP]', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = () => NO_WIP;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text without --json when no WIP plans', async () => {
      mockReaddirSyncFn = () => [];
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('no [WIP] plans')));
      assert.equal(process.exitCode, 0);
    });

    it('filters readFileSync errors in wipPlans filter', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md'];
      mockReadFileSyncFn = () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
    });

    it('filters non-.plans.md files from readdirSync', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'README.md', 'data.txt'];
      mockReadFileSyncFn = () => NO_WIP;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
    });
  });

  describe('with WIP plans via --plan', () => {
    it('returns pass=true when specialist review evidence found', async () => {
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 1);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, false);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=true when TRIVIAL exemption marker found', async () => {
      mockReadFileSyncFn = () => WIP_WITH_TRIVIAL;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, true);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=false when no validation evidence section', async () => {
      mockReadFileSyncFn = () => WIP_NO_EVIDENCE;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no validation evidence section found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('returns pass=false when no specialist review markers found', async () => {
      mockReadFileSyncFn = () => WIP_NO_MARKERS;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no specialist review evidence found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('scans every validation-evidence region, not only the first one', async () => {
      // The reviewed step's evidence block is NOT the first one in a long
      // multi-step plan, so a leftmost-match scan would miss it.
      mockReadFileSyncFn = () => WIP_TWO_EVIDENCE_SECTIONS;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(process.exitCode, 0);
    });

    it('returns found=false when file not readable on second call', async () => {
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return '[WIP]\n'; // filter call
        throw new Error('ENOENT'); // checkPlan call
      };
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.planResults[0].reason, 'file not readable');
    });

    it('returns found=false when content no longer has [WIP] on second call', async () => {
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return '[WIP]\n'; // filter call
        return 'No WIP here.\n'; // checkPlan call
      };
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.planResults[0].reason, 'no [WIP] sections');
    });
  });

  describe('canonical `## Latest validation evidence` heading', () => {
    it('returns pass=true when the canonical heading carries specialist evidence', async () => {
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 1);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'specialist review evidence found in the validation evidence section',
      );
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=false when the canonical heading has no specialist markers', async () => {
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING_NO_MARKERS;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no specialist review evidence found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('returns pass=true when the canonical heading carries a TRIVIAL marker', async () => {
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING_TRIVIAL;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, true);
      assert.equal(process.exitCode, 0);
    });

    it('treats an empty canonical heading as no validation evidence section', async () => {
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING_EMPTY;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no validation evidence section found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('keeps `###` sub-blocks inside the canonical `##` region', async () => {
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING_SUBBLOCK;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(process.exitCode, 0);
    });

    it('ignores a canonical heading quoted inside a fenced code block', async () => {
      mockReadFileSyncFn = () => WIP_HEADING_INSIDE_CODE_FENCE;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no validation evidence section found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('scans every canonical section, not only the first one', async () => {
      mockReadFileSyncFn = () => WIP_TWO_CANONICAL_SECTIONS;
      const logs = await importGate(['--json', '--plan=plans/test.plans.md']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(process.exitCode, 0);
    });
  });

  describe('stale parseArgs default plan', () => {
    it('ignores the parseArgs default plan when --plan is absent', async () => {
      // `parseArgs` injects `plans/completed/Agentic_Workflow_Architecture.plans.md`
      // as its default. The gate must NOT validate that stale artifact —
      // it must scan the live plans/ directory instead.
      mockReaddirSyncFn = () => ['active.plans.md'];
      mockReadFileSyncFn = () => WIP_CANONICAL_HEADING;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 1);
      assert.equal(parsed.evidence.planResults[0].plan, 'active.plans.md');
      assert.equal(process.exitCode, 0);

      const readPaths = Object.keys(mockReadFileCalls);
      assert.ok(
        !readPaths.some((p) => p.includes('Agentic_Workflow_Architecture')),
        `stale default plan was read: ${readPaths.join(', ')}`,
      );
    });

    it('reports the live plan, not the stale default, when the live plan lacks evidence', async () => {
      mockReaddirSyncFn = () => ['active.plans.md'];
      mockReadFileSyncFn = () => WIP_NO_EVIDENCE;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.planResults[0].plan, 'active.plans.md');
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no validation evidence section found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('honors an explicit --plan over the stale default', async () => {
      mockReaddirSyncFn = () => ['active.plans.md'];
      mockReadFileSyncFn = (p) => {
        if (String(p).includes('explicit')) return WIP_CANONICAL_HEADING;
        return STALE_DEFAULT_PLAN;
      };
      const logs = await importGate([
        '--json',
        '--plan=plans/explicit.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 1);
      assert.equal(parsed.evidence.planResults[0].plan, 'explicit.plans.md');
      assert.equal(process.exitCode, 0);
    });
  });

  describe('with WIP plans via directory scan', () => {
    it('returns pass=true when all WIP plans have specialist review', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 2);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=false when some WIP plans lack specialist review', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      let callCount = 0;
      mockReadFileSyncFn = () => {
        callCount++;
        // Calls 1-2: filter (both WIP). Calls 3-4: checkPlan.
        if (callCount <= 2) return WIP_WITH_SPECIALIST;
        return WIP_NO_EVIDENCE;
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });
  });

  describe('text output', () => {
    it('emits PASS text when all WIP plans have specialist review', async () => {
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate(['--plan=plans/test.plans.md']);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('specialist-review gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with plan listing and fixHint when pass=false', async () => {
      mockReadFileSyncFn = () => WIP_NO_EVIDENCE;
      const logs = await importGate(['--plan=plans/test.plans.md']);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('test.plans.md')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('covers if(!r.found) false branch with mixed found results', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return WIP_WITH_SPECIALIST;
        const key = String(p);
        if (key.includes('plan1')) return WIP_WITH_SPECIALIST;
        return WIP_NO_EVIDENCE;
      };
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
    });
  });
});
