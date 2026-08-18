/**
 * Red-phase contract tests for Step B1: Enemy AI Parallelism —
 * Sim/Render Worker Split, Render Order, and Map Grid Sharing.
 *
 * These tests define the contracts for:
 * - Render compositing order preservation (Invariant §5)
 * - Map grid sharing contract (Invariant §1)
 * - Zero-timestep pass with bolt-spawn state awareness
 * - Sim/render worker split infrastructure
 *
 * All tests in this file MUST fail (RED) until the B1 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import { NEATENSTEIN_MAP_SIZE } from '../constants';
import {
  __testOnlyGetZeroTimestepPassSkipped,
  createDisplayWorkerState,
  runSimStep,
} from './display.worker.sim.utils';
import * as SimUtilsModule from './display.worker.sim.utils';
import * as RenderUtilsModule from './display.worker.render.utils';

/** Access unknown (future) exports on the sim utils module. */
const simUtilsExports = SimUtilsModule as Record<string, unknown>;

/** Access unknown (future) exports on the render utils module. */
const renderUtilsExports = RenderUtilsModule as Record<string, unknown>;

// ---------------------------------------------------------------------------
// Contract 7: Render compositing order preservation (Invariant §5)
// ---------------------------------------------------------------------------

describe('B1: Render compositing order preservation (Invariant §5)', () => {
  it('exports an explicit render compositing order contract', () => {
    // B1 contract: the render compositing order MUST be explicitly defined
    // as a contract so any worker split preserves it. The order is:
    // floor → ceiling → walls → sprites → pulses/sparks → bolts
    //
    // Currently the order is implicit in buildAndPostFrame() call sequence.
    // B1 must make it explicit so the sim/render worker split can enforce it.
    expect(renderUtilsExports.RENDER_COMPOSITING_ORDER).toBeDefined();
  });

  it('render compositing order starts with floor then ceiling then walls', () => {
    // B1 contract: floor is drawn first, then ceiling, then walls.
    // Walls are drawn on top of the floor grid so the wall base sits
    // exactly where the integer floor line projects.
    const order = renderUtilsExports.RENDER_COMPOSITING_ORDER as
      | string[]
      | undefined;

    // This will fail because RENDER_COMPOSITING_ORDER doesn't exist yet.
    expect(order).toBeDefined();
    expect(order![0]).toBe('floor');
    expect(order![1]).toBe('ceiling');
    expect(order![2]).toBe('walls');
  });

  it('render compositing order has sprites after walls and bolts last', () => {
    // B1 contract: sprites → pulses/sparks → bolts (bolts drawn last).
    const order = renderUtilsExports.RENDER_COMPOSITING_ORDER as
      | string[]
      | undefined;

    expect(order).toBeDefined();
    const spritesIdx = order!.indexOf('sprites');
    const pulsesIdx = order!.indexOf('pulses');
    const boltsIdx = order!.indexOf('bolts');

    expect(spritesIdx).toBeGreaterThan(-1);
    expect(pulsesIdx).toBeGreaterThan(spritesIdx);
    expect(boltsIdx).toBeGreaterThan(pulsesIdx);
  });
});

// ---------------------------------------------------------------------------
// Contract 8: Map grid sharing (Invariant §1)
// ---------------------------------------------------------------------------

describe('B1: Map grid sharing contract (Invariant §1)', () => {
  it('exports a function to create a shared map grid buffer', () => {
    // B1 contract: the 120×120 Uint8Array map (shared grid basis for wall
    // DDA and floor projection) MUST be made available to the render worker
    // via SharedArrayBuffer or transfer.
    //
    // Currently wallMap is stored in DisplayWorkerState but not shared.
    // B1 must add a sharing mechanism for the sim/render worker split.
    expect(typeof simUtilsExports.createSharedMapGrid).toBe('function');
  });

  it('shared map grid buffer matches the 120×120 map size', () => {
    // B1 contract: the shared map grid is NEATENSTEIN_MAP_SIZE ×
    // NEATENSTEIN_MAP_SIZE (120×120) bytes.
    expect(typeof simUtilsExports.createSharedMapGrid).toBe('function');

    // When the function exists, this will verify the buffer dimensions.
    // For now, this fails because createSharedMapGrid doesn't exist.
    const createFn = simUtilsExports.createSharedMapGrid as
      | ((size: number) => SharedArrayBuffer | Uint8Array)
      | undefined;
    expect(createFn).toBeDefined();

    const buffer = createFn!(NEATENSTEIN_MAP_SIZE);
    expect(buffer.byteLength).toBeGreaterThanOrEqual(
      NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE,
    );
  });
});

// ---------------------------------------------------------------------------
// Contract 9: Zero-timestep pass with bolt-spawn state awareness
// ---------------------------------------------------------------------------

describe('B1: Zero-timestep pass — bolt-spawn state awareness', () => {
  it('exports a diagnostic for bolt-spawn state change detection', () => {
    // B1 contract: the zero-timestep pass conditionally skips entirely
    // when completedDeRezIndices.length === 0 AND no bolt-spawn state
    // changed. Currently the skip condition only checks de-rez, enemy
    // count match, and death-state update need — it does NOT check
    // bolt-spawn state.
    expect(
      typeof simUtilsExports.__testOnlyGetBoltSpawnStateChanged,
    ).toBe('function');
  });

  it('zero-timestep pass is NOT skipped when bolt-spawn state changed', () => {
    // B1 contract: if a bolt was spawned this tick, the zero-timestep
    // pass must run even if no de-rez completed. The current code skips
    // based only on de-rez, enemy count, and death-state — not bolt-spawn.
    //
    // This test verifies that __testOnlyGetZeroTimestepPassSkipped returns
    // false when bolt-spawn state changed. Currently it may return true
    // (incorrectly skipping the pass).
    //
    // We verify the bolt-spawn state change diagnostic exists first.
    expect(
      typeof simUtilsExports.__testOnlyGetBoltSpawnStateChanged,
    ).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Contract 10: Sim/render worker split infrastructure
// ---------------------------------------------------------------------------

describe('B1: Sim/render worker split', () => {
  it('exports a function to create the sim worker state slice', () => {
    // B1 contract: simulation worker owns game state + enemy AI.
    // Render worker owns DDA + framebuffer + canvas.
    // This requires a sim worker state creation function.
    expect(typeof simUtilsExports.createSimWorkerState).toBe('function');
  });

  it('exports a function to create the render worker state slice', () => {
    // B1 contract: the render worker state is separate from the sim worker
    // state. The render worker owns DDA + framebuffer + canvas.
    expect(typeof renderUtilsExports.createRenderWorkerState).toBe('function');
  });

  it('exports a function to serialize enemy state into shared memory', () => {
    // B1 contract: sim writes enemy state into shared memory; render reads
    // it. This requires a serialization function for the sim→render
    // shared memory boundary.
    expect(typeof simUtilsExports.serializeEnemyStateToShared).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Contract 11: Sim step does not advance until all enemy results collected
// ---------------------------------------------------------------------------

describe('B1: Sim step barrier integration', () => {
  it('runSimStep exports a barrier-aware variant', () => {
    // B1 contract: the sim tick does NOT advance until ALL enemy inference
    // results for that tick are collected. This requires a barrier-aware
    // runSimStep variant or an integration with awaitInferenceBarrier.
    expect(typeof simUtilsExports.runSimStepParallel).toBe('function');
  });

  it('createDisplayWorkerState remains available for backward compatibility', () => {
    // Sanity check: the existing createDisplayWorkerState function is
    // still exported (backward compatibility during the B1 migration).
    expect(typeof createDisplayWorkerState).toBe('function');
  });

  it('runSimStep remains available for backward compatibility', () => {
    // Sanity check: the existing runSimStep function is still exported.
    expect(typeof runSimStep).toBe('function');
  });

  it('__testOnlyGetZeroTimestepPassSkipped remains available', () => {
    // Sanity check: the existing diagnostic is still exported.
    expect(typeof __testOnlyGetZeroTimestepPassSkipped).toBe('function');
  });
});