/**
 * Red tests for the acceleration observer and report surface.
 *
 * These tests define the expected public API of `acceleration.observer.ts`
 * before it is implemented. They will fail with TS2307 "Cannot find module"
 * until `src/acceleration/acceleration.observer.ts` exists.
 *
 * Per the plan, `AccelerationObserver` is an injectable callback surface,
 * `NoopAccelerationObserver` is the safe empty default, and `AccelerationReport`
 * is the rolling in-memory report shape consumed by callers.
 */

import {
  createBackendCacheObserver,
  NoopAccelerationObserver,
  type AccelerationObserver,
  type AccelerationReport,
} from './acceleration.observer';

describe('acceleration.observer', () => {
  describe('NoopAccelerationObserver', () => {
    it('is a safe empty default observer', () => {
      expect(NoopAccelerationObserver).toEqual({});
    });
  });

  describe('createBackendCacheObserver', () => {
    it('returns null before any backend change event', () => {
      const { getBackend } = createBackendCacheObserver();

      expect(getBackend()).toBeNull();
    });

    it('caches the current backend from an onBackendChange event', () => {
      const { observer, getBackend } = createBackendCacheObserver();

      observer.onBackendChange!({
        previous: null,
        current: 'gpu',
        reason: 'test',
        timestamp: Date.now(),
      });

      expect(getBackend()).toBe('gpu');
    });

    it('replaces the cached backend on a subsequent change event', () => {
      const { observer, getBackend } = createBackendCacheObserver();

      observer.onBackendChange!({
        previous: null,
        current: 'gpu',
        reason: 'test',
        timestamp: Date.now(),
      });
      observer.onBackendChange!({
        previous: 'gpu',
        current: 'cpu',
        reason: 'test',
        timestamp: Date.now(),
      });

      expect(getBackend()).toBe('cpu');
    });
  });

  describe('AccelerationObserver', () => {
    it('can be implemented with an onFallback callback', () => {
      const observer: AccelerationObserver = {
        onFallback: jest.fn(),
      };

      expect(observer.onFallback).toBeDefined();
    });

    it('can be implemented with an onTelemetry callback', () => {
      const observer: AccelerationObserver = {
        onTelemetry: jest.fn(),
      };

      expect(observer.onTelemetry).toBeDefined();
    });

    it('can be implemented with an onBackendChange callback', () => {
      const observer: AccelerationObserver = {
        onBackendChange: jest.fn(),
      };

      expect(observer.onBackendChange).toBeDefined();
    });
  });

  describe('AccelerationReport', () => {
    it('tracks time spent in each backend', () => {
      const report: AccelerationReport = {
        timeInBackendMs: { cpu: 0, gpu: 0, worker: 0 },
        transitions: [],
        fallbackCount: 0,
        telemetry: [],
      };

      expect(report.timeInBackendMs).toEqual({ cpu: 0, gpu: 0, worker: 0 });
    });

    it('records backend transitions', () => {
      const report: AccelerationReport = {
        timeInBackendMs: { cpu: 0, gpu: 0, worker: 0 },
        transitions: [],
        fallbackCount: 0,
        telemetry: [],
      };

      expect(report.transitions).toEqual([]);
    });

    it('counts fallback events', () => {
      const report: AccelerationReport = {
        timeInBackendMs: { cpu: 0, gpu: 0, worker: 0 },
        transitions: [],
        fallbackCount: 0,
        telemetry: [],
      };

      expect(report.fallbackCount).toBe(0);
    });

    it('stores telemetry events', () => {
      const report: AccelerationReport = {
        timeInBackendMs: { cpu: 0, gpu: 0, worker: 0 },
        transitions: [],
        fallbackCount: 0,
        telemetry: [],
      };

      expect(report.telemetry).toEqual([]);
    });
  });
});
