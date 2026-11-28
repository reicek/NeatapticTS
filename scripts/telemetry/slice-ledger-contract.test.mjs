/**
 * @module slice-ledger-contract.test
 * @description RED contract home for the per-slice telemetry ledger.
 *
 * Slice: OT-S1-1 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home:
 *   - AC-003-A (the forbidden-field contract)
 *   - AC-004-A (the availability contract)
 *   - AC-002-A (the one-ledger-per-slice contract)
 *
 * The subject under test is `scripts/telemetry/telemetry-contract.mjs`, which the
 * GREEN lane (OT-S1-2) implements. This home resolves it LAZILY inside every
 * assertion, so an unimplemented contract is an ASSERTION FAILURE that names the
 * missing contract - never an import failure that would abort the whole file.
 *
 * THE CONTRACT SURFACE THIS HOME DEMANDS (OT-S1-2 must export exactly these):
 *   SCHEMA_VERSION                 - the integer schema version (1).
 *   REQUIRED_LEDGER_KEYS           - the key set the persisted artifact declares.
 *   FORBIDDEN_KEYS                 - the forbidden key set (R-3).
 *   FORBIDDEN_VALUE_PATTERNS       - the forbidden value patterns (R-3).
 *   LEDGER_KEY_FIELDS              - the ledger key fields: plan_id and slice_id (R-1).
 *   LEDGER_WRITER_PATH             - the repo-relative path of the DONE writer (R-8).
 *   TOOL_EXECUTION_BUCKETS         - the five safe buckets (R-5).
 *   scanPersistedText(text)        - the ONE scanner; the offending key or pattern.
 *   emitLedgerBlock(block)         - the deterministic emitter; byte-identical (R-2).
 *   validateUsageCounter(counter)  - the availability validator (R-4).
 *   demandsLedger({ slice_id, pre_existing_done_slice_ids }) - forward-only (R-7).
 *
 * THE LANES ASSERT THE CONTRACT, not an implementation detail. Two tolerances are
 * deliberate and are documented where they are used:
 *   - the scanner's violation shape and the validator's verdict shape are normalised,
 *     because the tracker pins the CONTENT of the contract, not its descriptor;
 *   - the unavailability reason is accepted as `reason` (R-4) or as `unavailable_reason`
 *     (the canonical schema-v1 example), because the tracker uses both spellings.
 *
 * THE LEDGER WRITER IS PINNED BY ITS DECLARED PATH, NOT BY ITS EXISTENCE ON DISK:
 * `scripts/telemetry/write-slice-ledger.mjs` is owned by `OT-S3-3` (Phase 3), so a
 * lane that demanded the file to exist would keep this step's GREEN lane (OT-S1-5)
 * permanently red. AC-002-A is therefore pinned through `LEDGER_WRITER_PATH`,
 * `LEDGER_KEY_FIELDS` and the emitter's idempotency, which are the contract facts
 * the writer must honour.
 */

import { describe, it, expect } from '@jest/globals';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const PLANS_DIR = path.join(REPO_ROOT, 'plans');

/**
 * Resolves the contract module. A module that cannot be resolved comes back as
 * a `missingContract` marker instead of an exception, so the failure is an
 * assertion failure that NAMES the missing contract.
 */
async function loadContractModule() {
  try {
    return await import('./telemetry-contract.mjs');
  } catch (error) {
    const message = String(error?.message ?? error);
    const isMissing =
      (error?.code === 'ERR_MODULE_NOT_FOUND' || /Cannot find module/i.test(message)) &&
      message.includes('telemetry-contract.mjs');
    const detail = isMissing
      ? 'the module file scripts/telemetry/telemetry-contract.mjs does not exist yet'
      : `the module file exists but failed to load (${message})`;
    return {
      missingContract: `scripts/telemetry/telemetry-contract.mjs is not implemented yet: ${detail}`,
    };
  }
}

/**
 * Resolves the contract module inside the calling lane and asserts that it
 * exists. Called from inside every `it()` block, never at module scope.
 */
async function requireContract() {
  const contract = await loadContractModule();
  expect(contract.missingContract).toBeUndefined();
  return contract;
}

/**
 * Normalises a declared collection (keys, buckets, patterns) into an array so a
 * lane asserts the CONTENT of the contract and not the container type.
 */
function asList(value) {
  if (Array.isArray(value)) return value;
  if (value instanceof Set) return [...value];
  if (value && typeof value === 'object') return Object.values(value);
  return [];
}

/**
 * Reads the scanner result as a list of violations. The contract requires the
 * scanner to report the offending key or pattern; the home does not pin the
 * exact descriptor shape.
 */
function violationsOf(scanResult) {
  if (scanResult === null || scanResult === undefined || scanResult === false) {
    return [];
  }
  if (Array.isArray(scanResult)) return scanResult;
  if (typeof scanResult === 'string') return [scanResult];
  if (typeof scanResult === 'object') {
    if (Array.isArray(scanResult.violations)) return scanResult.violations;
    if (scanResult.violation !== undefined) return [scanResult.violation];
    return [scanResult];
  }
  return [scanResult];
}

/**
 * Tells whether one forbidden value pattern matches a sample value. A pattern
 * may be a regular expression, a regex source or a `{ pattern }` record.
 */
function matchesForbiddenValue(contract, sample) {
  return asList(contract.FORBIDDEN_VALUE_PATTERNS).some((entry) => {
    if (entry instanceof RegExp) {
      return new RegExp(entry.source, entry.flags.replace('g', '')).test(sample);
    }
    if (typeof entry === 'string') {
      try {
        return new RegExp(entry).test(sample);
      } catch {
        return false;
      }
    }
    if (entry && typeof entry === 'object') {
      if (typeof entry.test === 'function') return Boolean(entry.test(sample));
      const source = entry.pattern ?? entry.regex ?? entry.source;
      if (typeof source !== 'string') return false;
      try {
        const flags = String(entry.flags ?? '').replace('g', '');
        return new RegExp(source, flags).test(sample);
      } catch {
        return false;
      }
    }
    return false;
  });
}

/**
 * Normalises a validator verdict. The contract module may report the verdict
 * as a boolean, as `{ valid }`, `{ ok }`, `{ errors }`, `{ violations }` or by
 * throwing; the home asserts the OUTCOME and not the descriptor shape.
 */
function isValidVerdict(result) {
  if (result === true) return true;
  if (result === false || result === null || result === undefined) return false;
  if (Array.isArray(result)) return result.length === 0;
  if (typeof result === 'object') {
    if (typeof result.valid === 'boolean') return result.valid;
    if (typeof result.ok === 'boolean') return result.ok;
    if (Array.isArray(result.errors)) return result.errors.length === 0;
    if (Array.isArray(result.violations)) return result.violations.length === 0;
    return false;
  }
  return Boolean(result);
}

/**
 * Runs the counter validator and reports the verdict.
 *
 * `present` distinguishes "the contract refuses this counter" from "the contract
 * does not export a validator at all", and `threw` distinguishes a refusal from a
 * crash. Without those two flags a NEGATIVE lane would pass vacuously against a
 * contract module that simply lacks the validator - the anti-vacuity requirement
 * of AC-003-GREEN.
 */
function counterVerdict(contract, counter) {
  if (typeof contract.validateUsageCounter !== 'function') {
    return {
      present: false,
      threw: false,
      valid: false,
      detail: 'the contract module does not export validateUsageCounter(counter)',
    };
  }
  try {
    const result = contract.validateUsageCounter(counter);
    return {
      present: true,
      threw: false,
      valid: isValidVerdict(result),
      detail: JSON.stringify(result ?? null),
    };
  } catch (error) {
    return {
      present: true,
      threw: true,
      valid: false,
      detail: String(error?.message ?? error),
    };
  }
}

/**
 * The canonical schema-v1 ledger block of the tracker, used as the fixture of
 * the emitter and of the availability lanes.
 */
function sampleLedgerBlock() {
  return {
    schema_version: 1,
    plan_id: 'opentelemetry-slice-telemetry',
    slice_id: 'OT-S3-2',
    status: '[DONE]',
    measurement_state: 'MEASURED',
    otel_state: 'DISABLED',
    duration_ms: 41233,
    runtime: {
      node_version: '22.14.0',
      heap_used_bytes: 41234567,
      rss_bytes: 98765432,
    },
    usage: {
      token_usage: {
        available: true,
        source: 'cortex-get-slice-context',
        measures: 'context tokens DELIVERED to the slice, not model consumption',
        context_tokens_delivered: 18432,
        prompt_tokens: null,
        completion_tokens: null,
        total_tokens: null,
        unavailable_reason: 'the repo has no access to the host model accounting',
      },
      tool_executions: {
        available: true,
        source: 'copilot-cli-posthook-observation-window',
        measures:
          'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
        scope: 'slice-dispatch-window',
        total: 17,
        completed_executions: 16,
        failed_executions: 1,
        sessions_observed: 1,
        breakdown: { read: 9, edit: 4, search: 4, exec: 0, other: 0 },
      },
    },
    validation: {
      total: 3,
      passed: 3,
      failed: 0,
      skipped: 0,
      results: [{ name: 'unit', status: 'passed', duration_ms: 412 }],
    },
  };
}

/**
 * An AVAILABLE tool-execution counter: it carries a source, a measures string
 * that states what it measures and a scope.
 */
function availableToolExecutionsCounter() {
  return {
    available: true,
    source: 'copilot-cli-posthook-observation-window',
    measures:
      'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
    scope: 'slice-dispatch-window',
    total: 17,
    completed_executions: 16,
    failed_executions: 1,
    sessions_observed: 1,
    breakdown: { read: 9, edit: 4, search: 4, exec: 0, other: 0 },
  };
}

/**
 * An UNAVAILABLE tool-execution counter: it carries a reason and NULL numeric
 * fields, never a zero. The reason is spelled `unavailable_reason` here because
 * that is the spelling of the canonical schema-v1 example; the lane that covers
 * R-4's `reason` spelling asserts that one separately.
 */
function unavailableToolExecutionsCounter() {
  return {
    available: false,
    source: 'copilot-cli-posthook-observation-window',
    measures:
      'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
    scope: 'slice-dispatch-window',
    total: null,
    completed_executions: null,
    failed_executions: null,
    sessions_observed: null,
    breakdown: null,
    unavailable_reason: 'the host CLI exposes no post-execution hook on this machine',
  };
}

/**
 * A persisted text that carries neither a forbidden key nor a forbidden value.
 */
const CLEAN_PERSISTED_TEXT = [
  'telemetry_ledger:',
  '  schema_version: 1',
  "  plan_id: 'opentelemetry-slice-telemetry'",
  "  slice_id: 'OT-S3-2'",
  "  status: '[DONE]'",
].join('\n');

/**
 * The slice ids the LIVE trackers already carry as DONE, measured on the live
 * bytes. The tracker format is a YAML list item, so the id is read from the SAME
 * line as the `- slice_id:` marker and the status is read from the block that
 * follows it.
 */
const TRACKER_FILE_PATTERN = /\.(plans|logs)\.md$/;
const SLICE_ID_LINE_PATTERN =
  /^[ \t]*-[ \t]+slice_id:[ \t]*['"]?([A-Za-z0-9][A-Za-z0-9._-]*)['"]?[ \t]*$/gm;
const DONE_STATUS_LINE_PATTERN = /^[ \t]*status:[ \t]*['"]?\[DONE\]/m;

/**
 * Every tracker file under `plans/`, recursively and in a stable order: the
 * active `plans/*.plans.md` trackers, their companion `.logs.md` files and the
 * archived `plans/completed/` trackers.
 */
function collectTrackerFiles() {
  const files = [];
  const pending = [PLANS_DIR];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        pending.push(fullPath);
      } else if (TRACKER_FILE_PATTERN.test(entry.name)) {
        files.push(fullPath);
      }
    }
  }
  return files.sort();
}

/**
 * Collects the slice ids the LIVE trackers already carry as DONE.
 */
function collectLiveDoneSliceIds() {
  const doneSliceIds = new Set();
  for (const file of collectTrackerFiles()) {
    const trackerText = readFileSync(file, 'utf8');
    const marks = [];
    SLICE_ID_LINE_PATTERN.lastIndex = 0;
    let match;
    while ((match = SLICE_ID_LINE_PATTERN.exec(trackerText)) !== null) {
      marks.push({ id: match[1], start: match.index, end: SLICE_ID_LINE_PATTERN.lastIndex });
    }
    for (let index = 0; index < marks.length; index += 1) {
      const blockEnd =
        index + 1 < marks.length ? marks[index + 1].start : trackerText.length;
      const block = trackerText.slice(marks[index].end, blockEnd);
      if (DONE_STATUS_LINE_PATTERN.test(block)) {
        doneSliceIds.add(marks[index].id);
      }
    }
  }
  return [...doneSliceIds].sort();
}

/**
 * THE PRE-EXISTING DONE SET - the activation marker of R-7 - MEASURED from the
 * live trackers and FROZEN here as a fixture.
 *
 * It is frozen on purpose. Re-measuring it on every run would silently turn the
 * forward-only rule into "every slice that happens to be DONE right now is
 * exempt", which contradicts R-7 ("the gate demands a ledger only for slices
 * that completed at or after the activation marker") as soon as a post-activation
 * slice completes. The authoritative frozen set is owned by `OT-S3-4`; this
 * fixture is the RED-time measurement, and a lane proves that every id in it is
 * still recorded as DONE in the live trackers, so the fixture is provable against
 * the live trackers rather than invented.
 */
const PRE_EXISTING_DONE_SLICE_IDS = `
  01-green 01-impl 01-red 02-barrel-coverage 02-browser-scenario 02-clarification-cap 02-create-snapshots 02-green
  02-green-activation-stream 02-green-scripts 02-host-overlay 02-lazy-facade 02-main-thread-plumb 02-move-db 02-move-models 02-move-proofs
  02-move-scripts 02-red-activation-stream 02-repo-cortex 02-runtime-hooks 02-worker-capture 02b-config 02b-gate-tests 02b-green
  02b-mcp-test 03-01 03-01-green 03-01-impl 03-01-red 03-02 03-02-green 03-02-impl
  03-02-red 03-03 03-03-green 03-03-impl 03-03-red 03-04 03-04-green 03-04-impl
  03-04-red 03-05 03-05-green 03-05-impl 03-05-red 03-06 03-06-green 03-06-impl
  03-06-red 03-07 03-child-spawns 03-green 03-green-init-schema 03-init-schema 03-model-dirs 03-red-tests
  03-runtime-context 03-snapshot-output 03-traceability-ids 03A-wire-growth-cycle 03B-green-validate-wiring 03C-redesign-generator-and-demo 03D-relocate-tests 04-a
  04-b 04-c 04-confirm-red 04-green 04-green-smoke 04-impl 04-impl-non-overlapping 04-impl-seed-policy
  04-obs-constants 04-pit-fields 04-red 04-red-tests 04-self-check 04-tasks-template 04-wire-reproduction 04-wire-vector
  05-gitignore 05-green 05-jest 05-package-json 05-spec-checklist 06-ammo-render 06-gates 06-green
  06-mcp-config 06-mcp-tools 06-skills-agents 07-bug-triage 07-green 07-register-plan 07-remove-old-tree 07-research-artifact
  07-robot-sprite-capability-tests 08-extension-catalog 09-verbatim-phrases 0a-collision-utils 0a-state-utils 0b-atlas-utils 0b-guards-utils 0c-band-utils
  0c-projection-utils 0c-shade-utils 10-fix-floorceiling-neon 10-fix-perf-throttle 10-fix-walls-fog 10-green 10-impl-encoded-sprite 10-impl-framebuffer
  10-impl-projection 10-learning-event-schema 10-red-encoded-sprite 10.2-fix-bolt-ai-spawn 10.2-fix-collision-sync 10.2-fix-sprite-facing-sort 10.2-fix-walk-anim-rendercap 10.2-green-visiblebrowser
  10.3-cull-enemies-30 10.3-floor-ceiling-cap 10.3-fog-cap-30 10.3-perf-cleanup 10.3-raf-clock 12-aspect-detail 12-green 12-red-gun
  12-voxel-sprite 19.4 1a-column-utils 1a-orchestrator 1a-projection-utils 1b-orchestrator 1c-bolt-utils 1c-declarative-draws
  20-green-validation 21-green-validation 21-red-tests 21-red-tests-evidence 22-core-metrics 22-green-validation 22-red-tests 23-core-ann
  23-green 23-green-validation 23-impl 23-red-tests 2a-bolt-utils 2a-input-pickup-impact-utils 2a-look-gun-time-collisionmap-utils 2b-color-dom-constants-utils
  2b-statusbar-humanmode-wave-utils 3a-fire-orchestrator 3a-move-collision-flank-utils 3a-spawn-death-stun-utils 3b-sprite-utils 3c-compare-utils 3c-png-voxel-utils 3d-navigation-utils
  4a-color-canvas-utils 4a-input-raycast-utils 4b-render-utils 4c-sim-autoai-utils 4c-state-accessors 5B-dynamic-delta-impl 5C-magnitude-scaling 5a-curriculum-utils
  5a-mlp-math-backprop-utils 5b-render-loop-utils 5c-bootstrap-utils 5c-canvas-dimensions-utils A1-green A1-red A1-red-tests A1-schema-enrichment
  A2-filter-params A2-fix2 A2-impl A2-red-tests A3-green A3-red A5-green A5-impl
  B1-green B1-impl B1-red B1-red-tests B1-tool-impl B2-impl B2-red B2-red-tests
  B3-green B3-impl B3-red B3-red-tests B4-apply-to-plan B4-green B4-impl B4-impl-validator
  B4-integ-green B4-red B5-green B5-impl B5-red B5-red-tests B6-green B6-impl
  B6-red B7-green B7-impl C1-gate-and-skill C1-green C1-impl C1-red C1-red-tests
  C2-agent-and-sample C2-green C2-impl C2-red C2-red-tests C3-design-rag-fix-packet C3-example-in-plan C3-green
  C3-impl C3-red C4-green C4-impl C4-red C4-red-tests C5-green C5-impl
  C5-red D1-green D1-impl D1-red-tests D2-concurrency-docs D2-green D2-impl D2-rag-exception-docs
  D2-red-tests D3-green D3-impl E1-green E1-impl E1-red-tests E2-green E2-impl
  E2-red-tests P0S1-green P0S1-sprite-redesign P0S2-accent-color P0S2-green P1-S1-1 P1-S1-2 P1-S1-3
  P1-S1-4 P1-S1-5 P1-S1-A P1-S1-B P1-S1-C P1-S1-D P1-S1-green P1-S2-1
  P1-S2-2 P1-S2-3 P1-S2-4 P1-S2-5 P1-S2-6 P1-S2-A P1-S2-B P1-S2-C
  P1-S2-green P1-S3-1 P1-S3-2 P1-S3-3 P1-S3-4 P1-S3-5 P1-S3-A P1-S3-B
  P1-S3-C P1-S3-D P1-S3-E P1-S3-G1 P1-S3-G2 P1-S4-A P1-S4-B P1-S4-C
  P1-S4-D P1-S4-E P1-S4-G1 P1-S4-G2 P1-S5-A P1-S5-B P1-S5-C P1-S5-D
  P1-S5-green P1-S6-A P1-S6-B P1-S6-C P1-S6-D P1-S6-E P1-S6-G1 P1-S6-G2
  P1-S7-A P1-S7-B P1-S7-C P1-S7-D P1-S7-E P1-S7-G1 P1-S7-G2 P1-ai-boot-race
  P1-ai-centerline P1-ai-centerline-green P1-ai-centerline-red P1-ai-racing-line P1-arcade-drift P1-arcade-drift-green P1-arcade-drift-model P1-arcade-drift-visual
  P1-boot-composition P1-boot-glue P1-boot-glue-green P1-car-shape P1-composition-green P1-correctness-fixup P1-correctness-green P1-crowd-green
  P1-crowd2-green P1-decoration-scale P1-drift-collision-live P1-drift-feel-green P1-drift-feel-red P1-drift-feel-tune-contracts P1-drift-feel-tune-model P1-drift-feel-tune-state
  P1-drift-law-code-green P1-drift-law-green P1-drift-law-proportional P1-drift-law-red P1-drift-law-repin P1-drift-presentation-car P1-drift-presentation-green P1-drift-presentation-hud
  P1-drift-presentation-red P1-drift-rewire-boot-mode P1-drift-rewire-green P1-drift-rewire-red P1-drift-rewire-road-surface P1-drift-rewire-travel-camera P1-drift-traction-scalar P1-drift-traction-seam
  P1-driftcam-green P1-drive-loop-live P1-editor-loop-green P1-editor-preview P1-frame-composer P1-grid-footprint P1-ground-composition P1-mechanic-crew
  P1-momentum-green P1-momentum-law P1-momentum-red P1-momentum-repin P1-momentum-service P1-neatenstein-public P1-outer-edge-geometry P1-outer-edge-green
  P1-outer-edge-paint P1-outer-edge-red P1-outer-edge-repin P1-physics-drive P1-prop-clearance-props P1-prop-clearance-racingline P1-prop-clearance-red P1-prop-clearance-repin
  P1-raster-unclip-green P1-raster-unclip-impl P1-raster-unclip-red P1-raster-unclip-repin P1-right-world-fidelity-green P1-right-world-green P1-right-world-props P1-road-edge-hex
  P1-sprite-directional-walk-look P1-sprite-elevation P1-sprite-perspective-scale P1-sprite-scale-green P1-steer-animation P1-track-outer-corner-markers-green P1-tron-bleachers P1-turn-drift-green
  P1-turnrate-ai P1-turnrate-green P1-turnrate-law P1-turnrate-red P1-turnrate-repin P1-wheel-mount P2-S02-A P2-S02-B
  P2-S02-C P2-S02-D P2-S1-A P2-S1-B P2-S1-C P2-S1-green P2-S4-1 P2-S4-2
  P2-S4-3 P2-S4-4 P2-S5-1 P2-S5-2 P2-S5-3 P2-S6-1 P2-S6-2 P2-S6-3
  P2-S6-4 P2S1-gen-sync P2S2-enemy-mlp-inject P2S3-01 P2S3-02 P2S3-03 P2S3-04 P2S3-05
  P2S3-06 P3-S02-A P3-S02-B P3-S02-C P3-S02-D P3S1-advance-wave P3S2-arms-race-cadence P3S3-01
  P3S3-02 P3S3-03 P3S3-04 P3S3-05 P3S3-06 P3S3-07 P3S3-08 P3S3-09
  P4-S02-A P4-S02-B P4-S02-C P4-S1-green P4S1-worker-branch P4S2-sensor-activation P4S3-01 P4S3-02
  P4S3-03 P4S3-04 P4S3-05 P4S3-06 P4S3-07 P4S3-08 P4S3-09 P4S3-10
  P4S3-11 P4S3-12 P5S1-real-episode P5S2a-telemetry P5S2b-fitness-cadence P5S2c-bfs-pooling P5S3-01 P5S3-02
  P5S3-03 P6S3-01 P6S3-02 P6S3-03 P6S3-04 P6S3-05 P6S3-06 P7S3-01
  P7S3-02 P7S3-03 P7S4-01 P7S4-02 P7S4-03 P8S1-coverage-closure P8S1-green P8S3-01
  P8S3-02 P8S3-03 P8S3-04 P8S3-05 P8S3-06 P8S3-07 P8S3-08 P8S3-09
  p1-02-green p1-02-impl p1-02-red p1-03-green p1-03-impl p1-03-red p2-04-green p2-04-impl-guiding-lines
  p2-04-red-guiding-lines p8-s08-green p8-s08-impl p8-s08-red p8-s14-green p8-s14-impl p8-s14-red p8-s15-green
  p8-s15-impl-fix p8-s15-impl-fix-lod p8-s15-impl-hover p8-s15-impl-lod p8-s15-red p8-s16-green p8-s16-impl-env p8-s16-impl-renderer
  p8-s16-impl-worker p8-s16-impl-worker-fix p8-s16-red p8-s17-green p8-s17-impl p8-s17-red p8-s18-green p8-s18-impl-harness
  p8-s18-impl-pits p8-s18-red p8-s19-green p8-s19-impl-adaptation p8-s19-impl-browser p8-s19-red p8-s20-green p8-s20-impl-core
  p8-s20-impl-structure p8-s20-red p8-s21-green p8-s21-impl-evaluator p8-s21-impl-promotion p8-s21-red p8-s22-green step-5-green
`
  .trim()
  .split(/\s+/);

describe('slice-ledger contract', () => {
  describe('AC-003-A - the forbidden-field contract', () => {
    it('declares schema version 1 for the persisted ledger block', async () => {
      const contract = await requireContract();
      expect(contract.SCHEMA_VERSION).toBe(1);
    });

    it('declares every timestamp-bearing key as a forbidden key', async () => {
      const contract = await requireContract();
      expect(asList(contract.FORBIDDEN_KEYS)).toEqual(
        expect.arrayContaining([
          'timestamp',
          'time',
          'startTime',
          'endTime',
          'timeUnixNano',
          'observedTimestamp',
          'startedAt',
          'completedAt',
          'date',
          'datetime',
          'epoch',
        ]),
      );
    });

    it('declares the required ledger keys and none of them is a forbidden key', async () => {
      const contract = await requireContract();
      const requiredKeys = asList(contract.REQUIRED_LEDGER_KEYS);
      const forbiddenKeys = asList(contract.FORBIDDEN_KEYS);
      expect(requiredKeys).toEqual(
        expect.arrayContaining([
          'schema_version',
          'plan_id',
          'slice_id',
          'status',
          'measurement_state',
          'otel_state',
          'duration_ms',
          'runtime',
          'usage',
          'validation',
        ]),
      );
      expect(requiredKeys.filter((key) => forbiddenKeys.includes(key))).toEqual([]);
    });

    it('keeps the ledger key fields keyed on plan_id and slice_id and free of forbidden keys', async () => {
      const contract = await requireContract();
      const ledgerKeyFields = asList(contract.LEDGER_KEY_FIELDS);
      const forbiddenKeys = asList(contract.FORBIDDEN_KEYS);
      expect(ledgerKeyFields).toEqual(expect.arrayContaining(['plan_id', 'slice_id']));
      expect(ledgerKeyFields.filter((key) => forbiddenKeys.includes(key))).toEqual([]);
    });

    it('declares a non-empty set of forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(asList(contract.FORBIDDEN_VALUE_PATTERNS).length).toBeGreaterThan(0);
    });

    it('rejects an ISO-8601 datetime with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, '2026-11-26T20:57:33.321Z')).toBe(true);
    });

    it('rejects a calendar date with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, '2026-11-26')).toBe(true);
    });

    it('rejects an epoch-millisecond literal with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, '1764147453321')).toBe(true);
    });

    it('rejects an absolute Windows path with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, 'C:\\Users\\alice\\project\\file.ts')).toBe(
        true,
      );
    });

    it('rejects an absolute POSIX user path with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, '/Users/alice/project/file.ts')).toBe(true);
      expect(matchesForbiddenValue(contract, '/home/alice/project/file.ts')).toBe(true);
    });

    it('rejects an e-mail address with the forbidden value patterns', async () => {
      const contract = await requireContract();
      expect(matchesForbiddenValue(contract, 'alice@example.com')).toBe(true);
    });

    it('reports the offending key when a persisted text carries a timestamp key', async () => {
      const contract = await requireContract();
      expect(typeof contract.scanPersistedText).toBe('function');
      const violations = violationsOf(contract.scanPersistedText('  timestamp: 1764147453321'));
      expect(violations.length).toBeGreaterThan(0);
      expect(JSON.stringify(violations)).toContain('timestamp');
    });

    it('reports no violation for a persisted text that carries neither a forbidden key nor a forbidden value', async () => {
      const contract = await requireContract();
      expect(typeof contract.scanPersistedText).toBe('function');
      expect(violationsOf(contract.scanPersistedText(CLEAN_PERSISTED_TEXT))).toHaveLength(0);
    });

    it('emits a ledger block that carries neither a forbidden key nor a forbidden value', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(sampleLedgerBlock());
      expect(violationsOf(contract.scanPersistedText(emitted))).toHaveLength(0);
    });

    it('never persists a forbidden timestamp key inside a ledger block', async () => {
      const contract = await requireContract();
      const block = { ...sampleLedgerBlock(), timestamp: 1764147453321 };
      let persisted = '';
      try {
        persisted = String(contract.emitLedgerBlock(block));
      } catch {
        persisted = '';
      }
      const carriesTheKey = /^[ \t]*-?[ \t]*['"]?timestamp['"]?[ \t]*:/m.test(persisted);
      const detected =
        !carriesTheKey ||
        JSON.stringify(violationsOf(contract.scanPersistedText(persisted))).includes(
          'timestamp',
        );
      expect(detected).toBe(true);
    });
  });

  describe('AC-004-A - the availability contract', () => {
    it('accepts an available counter that carries a source and a measures string', async () => {
      const contract = await requireContract();
      const verdict = counterVerdict(contract, availableToolExecutionsCounter());
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(true);
    });

    it('accepts an unavailable counter whose numeric fields are null and which carries a reason', async () => {
      const contract = await requireContract();
      const verdict = counterVerdict(contract, unavailableToolExecutionsCounter());
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(true);
    });

    it('accepts an unavailable counter that spells the reason as `reason`', async () => {
      const contract = await requireContract();
      const counter = unavailableToolExecutionsCounter();
      counter.reason = counter.unavailable_reason;
      delete counter.unavailable_reason;
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(true);
    });

    it('refuses an unavailable counter whose numeric field is zero', async () => {
      const contract = await requireContract();
      const counter = { ...unavailableToolExecutionsCounter(), total: 0 };
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(false);
    });

    it('refuses an available counter that carries no source', async () => {
      const contract = await requireContract();
      const counter = availableToolExecutionsCounter();
      delete counter.source;
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(false);
    });

    it('refuses an available counter that carries no measures string', async () => {
      const contract = await requireContract();
      const counter = availableToolExecutionsCounter();
      delete counter.measures;
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(false);
    });

    it('refuses an available counter whose measures string is empty', async () => {
      const contract = await requireContract();
      const counter = { ...availableToolExecutionsCounter(), measures: '' };
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(false);
    });

    it('refuses an unavailable counter that carries no reason', async () => {
      const contract = await requireContract();
      const counter = unavailableToolExecutionsCounter();
      delete counter.reason;
      delete counter.unavailable_reason;
      const verdict = counterVerdict(contract, counter);
      expect(verdict.present).toBe(true);
      expect(verdict.threw).toBe(false);
      expect(verdict.valid).toBe(false);
    });

    it('declares exactly the five safe tool-execution buckets', async () => {
      const contract = await requireContract();
      const buckets = asList(contract.TOOL_EXECUTION_BUCKETS);
      expect(buckets).toEqual(
        expect.arrayContaining(['read', 'edit', 'search', 'exec', 'other']),
      );
      expect(buckets).toHaveLength(5);
    });

    it('emits the tool-execution breakdown as a count per bucket and never as an argument list', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(sampleLedgerBlock());
      expect(emitted).toMatch(/['"]?read['"]?\s*:\s*9/);
      expect(emitted).toMatch(/['"]?edit['"]?\s*:\s*4/);
      expect(emitted).not.toMatch(/arguments/i);
    });

    it('names the tool-execution counter `tool_executions` and never `tool_calls`', async () => {
      const contract = await requireContract();
      const emitted = contract.emitLedgerBlock(sampleLedgerBlock());
      expect(emitted).toMatch(/tool_executions/);
      expect(emitted).not.toMatch(/tool_calls/);
    });

    it('never emits an unavailable counter as a zero', async () => {
      const contract = await requireContract();
      const block = sampleLedgerBlock();
      block.usage.tool_executions = unavailableToolExecutionsCounter();
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).not.toMatch(/['"]?total['"]?\s*:\s*0\b/);
      expect(emitted).toMatch(/['"]?(unavailable_reason|reason)['"]?\s*:/);
    });
  });

  describe('AC-002-A - the one-ledger-per-slice contract', () => {
    it('declares the ledger writer path', async () => {
      const contract = await requireContract();
      const writerPath = String(contract.LEDGER_WRITER_PATH ?? '')
        .replace(/\\/g, '/')
        .replace(/^\.\//, '');
      expect(writerPath).toBe('scripts/telemetry/write-slice-ledger.mjs');
    });

    it('resolves the declared ledger writer path inside scripts/telemetry and never under plans', async () => {
      const contract = await requireContract();
      const writerPath = String(contract.LEDGER_WRITER_PATH ?? '');
      const resolved = path.resolve(REPO_ROOT, writerPath);
      const telemetryDir = path.join(REPO_ROOT, 'scripts', 'telemetry');
      const plansDir = path.join(REPO_ROOT, 'plans');
      expect(resolved.startsWith(`${telemetryDir}${path.sep}`)).toBe(true);
      expect(resolved.startsWith(`${plansDir}${path.sep}`)).toBe(false);
      expect(path.isAbsolute(resolved)).toBe(true);
    });

    it('keys the emitted ledger block on the plan id and the slice id', async () => {
      const contract = await requireContract();
      const block = sampleLedgerBlock();
      const emitted = contract.emitLedgerBlock(block);
      expect(emitted).toContain(block.plan_id);
      expect(emitted).toContain(block.slice_id);
    });

    it('emits byte-identical text for the same block, so a rewrite is idempotent', async () => {
      const contract = await requireContract();
      const first = contract.emitLedgerBlock(sampleLedgerBlock());
      const second = contract.emitLedgerBlock(sampleLedgerBlock());
      expect(second).toBe(first);
    });

    it('does not demand a ledger for a slice that completed before the activation marker', async () => {
      const contract = await requireContract();
      // The exemption is provable against the LIVE trackers: every frozen id is
      // still recorded as DONE there, so the frozen fixture is measured and not
      // invented.
      const liveDoneSliceIds = new Set(collectLiveDoneSliceIds());
      const noLongerDone = PRE_EXISTING_DONE_SLICE_IDS.filter(
        (sliceId) => !liveDoneSliceIds.has(sliceId),
      );
      expect(noLongerDone).toEqual([]);
      const demanded = PRE_EXISTING_DONE_SLICE_IDS.filter((sliceId) =>
        contract.demandsLedger({
          slice_id: sliceId,
          pre_existing_done_slice_ids: PRE_EXISTING_DONE_SLICE_IDS,
        }),
      );
      expect(demanded).toEqual([]);
    });

    it('demands a ledger for a slice that completed after the activation marker', async () => {
      const contract = await requireContract();
      expect(
        contract.demandsLedger({
          slice_id: 'OT-S1-2',
          pre_existing_done_slice_ids: PRE_EXISTING_DONE_SLICE_IDS,
        }),
      ).toBe(true);
    });
  });
});
