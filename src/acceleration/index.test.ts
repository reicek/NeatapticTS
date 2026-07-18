/**
 * Smoke tests for the public acceleration barrel.
 *
 * These tests only verify that the barrel re-exports the foundational symbols
 * expected by consumers of the acceleration layer.
 */

import {
  resolveAccelerationConfig,
  detectAcceleration,
  resolveAccelerationMode,
  AccelerationPolicy,
  LifecycleAccelerationPolicy,
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  NoopAccelerationObserver,
  type AccelerationMode,
  type AccelerationObserver,
  type AccelerationReport,
} from './index';

describe('acceleration barrel', () => {
  it('re-exports the acceleration config builder', () => {
    expect(resolveAccelerationConfig).toBeInstanceOf(Function);
  });

  it('re-exports acceleration constants', () => {
    expect(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD).toBe(1_024);
  });

  it('re-exports the noop observer', () => {
    expect(NoopAccelerationObserver).toEqual({});
  });

  it('re-exports acceleration type symbols as types', () => {
    const mode: AccelerationMode = 'cpu';
    expect(mode).toBe('cpu');
  });

  it('allows constructing an observer from the barrel type', () => {
    const observer: AccelerationObserver = {
      onFallback: jest.fn(),
    };

    expect(observer.onFallback).toBeDefined();
  });

  it('allows constructing a report from the barrel type', () => {
    const report: AccelerationReport = {
      timeInBackendMs: { cpu: 0, gpu: 0, worker: 0 },
      transitions: [],
      fallbackCount: 0,
      telemetry: [],
    };

    expect(report.fallbackCount).toBe(0);
  });

  it('re-exports detectAcceleration', () => {
    expect(detectAcceleration).toBeInstanceOf(Function);
  });

  it('re-exports resolveAccelerationMode', () => {
    expect(resolveAccelerationMode).toBeInstanceOf(Function);
  });

  it('re-exports AccelerationPolicy', () => {
    expect(new AccelerationPolicy()).toBeInstanceOf(AccelerationPolicy);
  });

  it('re-exports LifecycleAccelerationPolicy', () => {
    expect(new LifecycleAccelerationPolicy()).toBeInstanceOf(
      LifecycleAccelerationPolicy,
    );
  });
});
