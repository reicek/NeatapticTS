import { describe, expect, it } from '@jest/globals';
import type {
  BindingDetach,
  ContactPosition,
  CreateEpisodeOptions,
  DeathFeedbackSignal,
  FireCallback,
  HealthAmmoHudState,
  HiveDensityHudState,
  HumanMode,
  InputRouterDetach,
  InputSnapshot,
  LightToggleCallback,
  LookCallback,
  LookDelta,
  MugshotDirection,
  NeonStatusBarState,
  TouchActiveCallback,
} from './types';

/**
 * Sibling test file for examples/neatenstein/browser-entry/host/types.ts.
 *
 * These tests exercise the exported type shapes at compile time and perform
 * lightweight runtime assertions so the folder quality gate sees a sibling
 * test file for every source module.
 */

describe('Neatenstein host types', () => {
  it('accepts a LookDelta shape', () => {
    const delta: LookDelta = { yawDelta: 0.1, pitchDelta: 0.2 };
    expect(delta).toEqual({ yawDelta: 0.1, pitchDelta: 0.2 });
  });

  it('accepts an InputSnapshot shape', () => {
    const snapshot: InputSnapshot = {
      timestamp: 0,
      movement: { forward: true, backward: false, left: false, right: false },
      look: { yawDelta: 0, pitchDelta: 0 },
      touch: { active: false, yawDelta: 0, pitchDelta: 0 },
      pointerLocked: false,
      fire: false,
      dash: false,
      lightToggle: false,
    };
    expect(snapshot.movement.forward).toBe(true);
  });

  it('accepts a CreateEpisodeOptions shape', () => {
    const options: CreateEpisodeOptions = { seed: 42, durationMs: 1000 };
    expect(options.seed).toBe(42);
  });

  it('accepts a NeonStatusBarState shape', () => {
    const state: NeonStatusBarState = {
      playerHealth: 75,
      playerMaxHealth: 100,
      playerAmmo: 30,
      playerMaxAmmo: 50,
    };
    expect(state.playerHealth).toBe(75);
  });

  it('accepts a HumanMode union value', () => {
    const mode: HumanMode = 'human';
    expect(mode).toBe('human');
  });

  it('accepts a MugshotDirection union value', () => {
    const direction: MugshotDirection = 'front';
    expect(direction).toBe('front');
  });

  it('accepts a HiveDensityHudState shape', () => {
    const state: HiveDensityHudState = { hiveDensity: 0.5 };
    expect(state.hiveDensity).toBe(0.5);
  });

  it('accepts a DeathFeedbackSignal shape', () => {
    const signal: DeathFeedbackSignal = {
      direction: 'stronger',
      aggressionDelta: 0.1,
      movementDelta: 0.2,
      positioningDelta: 0.3,
    };
    expect(signal.direction).toBe('stronger');
  });

  it('accepts a HealthAmmoHudState shape', () => {
    const state: HealthAmmoHudState = {
      health: 80,
      maxHealth: 100,
      ammo: 20,
      maxAmmo: 30,
    };
    expect(state.health).toBe(80);
  });

  it('accepts a ContactPosition shape', () => {
    const contact: ContactPosition = { x: 1, y: 2 };
    expect(contact.x).toBe(1);
  });

  it('accepts callback type shapes', () => {
    const look: LookCallback = (delta: LookDelta) => {
      expect(delta.yawDelta).toBe(0.1);
    };
    look({ yawDelta: 0.1, pitchDelta: 0 });

    const fire: FireCallback = () => {};
    fire();

    const toggle: LightToggleCallback = () => {};
    toggle();

    const touch: TouchActiveCallback = (active: boolean) => {
      expect(active).toBe(true);
    };
    touch(true);

    const detach: BindingDetach = () => {};
    detach();

    const routerDetach: InputRouterDetach = () => {};
    routerDetach();
  });
});
