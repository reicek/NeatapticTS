/**
 * @module trace-summarize.test
 * @description Unit tests for the trace summarization script.
 */

import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  jest,
} from '@jest/globals';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';

import {
  summarizeTrace,
  extractTraceEvents,
  computeMetrics,
  parseCliArgs,
  isCliEntryPoint,
  runCli,
} from './trace-summarize.mjs';

/**
 * Comprehensive fixture exercising every metric branch: main-thread detection,
 * CPU time, layout thrashing, JS execution, paint, memory peak, dropped frames,
 * long tasks, and edge cases (missing dur, missing ts, wrong phase, wrong pid/tid).
 */
const COMPREHENSIVE_EVENTS = [
  { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 0, dur: 5_000 },
  {
    ph: 'M',
    name: 'process_name',
    pid: 1,
    tid: 0,
    ts: 0,
    args: { name: 'Browser' },
  },
  {
    ph: 'M',
    name: 'thread_name',
    pid: 1,
    tid: 1,
    ts: 0,
    args: { name: 'CrBrowserMain' },
  },
  { ph: 'M', name: 'thread_name', pid: 1, tid: 3, ts: 0 },
  {
    ph: 'M',
    name: 'thread_name',
    pid: 1,
    tid: 2,
    ts: 0,
    args: { name: 'CrRendererMain' },
  },
  { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 1_000, dur: 60_000 },
  { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 2_000, dur: 20_000 },
  { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 3_000, dur: 10_000 },
  { ph: 'X', name: 'SomeEvent', pid: 1, tid: 2, ts: 4_000 },
  { ph: 'B', name: 'AsyncEvent', pid: 1, tid: 2, ts: 5_000 },
  { ph: 'X', name: 'RecalculateStyles', pid: 1, tid: 2, ts: 6_000, dur: 5_000 },
  { ph: 'X', name: 'Layout', pid: 1, tid: 2, ts: 7_000, dur: 3_000 },
  { ph: 'X', name: 'Layout', pid: 1, tid: 2, ts: 8_000, dur: 2_000 },
  { ph: 'X', name: 'Paint', pid: 1, tid: 2, ts: 9_000, dur: 1_000 },
  { ph: 'X', name: 'RunTask', pid: 2, tid: 2, ts: 1_000, dur: 30_000 },
  { ph: 'X', name: 'RunTask', pid: 1, tid: 99, ts: 1_000, dur: 40_000 },
  { ph: 'X', name: 'FunctionCall', pid: 1, tid: 2, ts: 11_000, dur: 15_000 },
  { ph: 'X', name: 'EvaluateScript', pid: 1, tid: 2, ts: 12_000, dur: 8_000 },
  { ph: 'X', name: 'FunctionCall', pid: 1, tid: 2, ts: 13_000 },
  { ph: 'B', name: 'FunctionCall', pid: 1, tid: 2, ts: 14_000 },
  { ph: 'X', name: 'Paint', pid: 1, tid: 2, ts: 15_000, dur: 1_000 },
  { ph: 'X', name: 'CompositeLayers', pid: 1, tid: 2, ts: 16_000, dur: 2_000 },
  {
    ph: 'C',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 17_000,
    args: { jsHeapSizeUsed: 10_000_000 },
  },
  {
    ph: 'C',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 18_000,
    args: { jsHeapSizeUsed: 15_000_000 },
  },
  {
    ph: 'C',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 19_000,
    args: { Snapshot: { jsHeapSizeUsed: 20_000_000 } },
  },
  {
    ph: 'C',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 20_000,
    args: { Snapshot: {} },
  },
  {
    ph: 'C',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 21_000,
    args: { other: 123 },
  },
  { ph: 'C', name: 'JSHeapUsedBytes', pid: 1, tid: 2, ts: 22_000 },
  {
    ph: 'C',
    name: 'OtherCounter',
    pid: 1,
    tid: 2,
    ts: 23_000,
    args: { jsHeapSizeUsed: 999 },
  },
  {
    ph: 'X',
    name: 'JSHeapUsedBytes',
    pid: 1,
    tid: 2,
    ts: 24_000,
    args: { jsHeapSizeUsed: 999 },
  },
  { ph: 'X', name: 'DroppedFrame', pid: 1, tid: 2, ts: 25_000, dur: 1_000 },
  { ph: 'X', name: 'DroppedFrame', pid: 1, tid: 2, ts: 26_000, dur: 1_000 },
  { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 27_000 },
  { ph: 'B', name: 'RunTask', pid: 1, tid: 2, ts: 28_000 },
  { ph: 'X', name: 'NoTimestamp', pid: 1, tid: 2, dur: 1_000 },
];

describe('trace-summarize', () => {
  let tempDir;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), 'trace-summarize-'));
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  describe('computeMetrics', () => {
    it('extracts all metric categories from a comprehensive trace fixture', () => {
      const result = computeMetrics(COMPREHENSIVE_EVENTS);
      expect(result).toEqual({
        eventCount: 35,
        cpuTimeMs: 135,
        layoutThrashingCount: 1,
        jsExecutionMs: 23,
        paintEventCount: 3,
        memoryPeakBytes: 20_000_000,
        droppedFrames: 2,
        longTasks16ms: 4,
        longTasks50ms: 1,
      });
    });

    it('falls back to all events when no main thread is detected', () => {
      const events = [
        { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 0, dur: 20_000 },
      ];
      const result = computeMetrics(events);
      expect(result.cpuTimeMs).toBe(20);
    });

    it('detects the legacy CrRendererMainThread name', () => {
      const events = [
        {
          ph: 'M',
          name: 'thread_name',
          pid: 1,
          tid: 2,
          ts: 0,
          args: { name: 'CrRendererMainThread' },
        },
        { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 0, dur: 20_000 },
        { ph: 'X', name: 'RunTask', pid: 1, tid: 99, ts: 0, dur: 50_000 },
      ];
      const result = computeMetrics(events);
      expect(result.cpuTimeMs).toBe(20);
    });

    it('returns zero metrics for an empty events array', () => {
      const result = computeMetrics([]);
      expect(result).toEqual({
        eventCount: 0,
        cpuTimeMs: 0,
        layoutThrashingCount: 0,
        jsExecutionMs: 0,
        paintEventCount: 0,
        memoryPeakBytes: 0,
        droppedFrames: 0,
        longTasks16ms: 0,
        longTasks50ms: 0,
      });
    });
  });

  describe('extractTraceEvents', () => {
    it('returns a bare array directly', () => {
      expect(extractTraceEvents('[{"ph":"X"}]')).toEqual([{ ph: 'X' }]);
    });

    it('extracts events from standard trace format', () => {
      expect(extractTraceEvents('{"traceEvents":[{"ph":"X"}]}')).toEqual([
        { ph: 'X' },
      ]);
    });

    it('extracts events from MCP result array', () => {
      expect(extractTraceEvents('{"result":[{"ph":"X"}]}')).toEqual([
        { ph: 'X' },
      ]);
    });

    it('extracts events from MCP result.traceEvents', () => {
      expect(
        extractTraceEvents('{"result":{"traceEvents":[{"ph":"X"}]}}'),
      ).toEqual([{ ph: 'X' }]);
    });

    it('returns empty array when MCP result has no traceEvents', () => {
      expect(extractTraceEvents('{"result":{}}')).toEqual([]);
    });

    it('returns empty array for an empty object', () => {
      expect(extractTraceEvents('{}')).toEqual([]);
    });
  });

  describe('summarizeTrace', () => {
    it('reads a gzip-compressed trace file', async () => {
      const tracePath = path.join(tempDir, 'trace.json.gz');
      const payload = JSON.stringify({
        traceEvents: [
          { ph: 'X', name: 'RunTask', pid: 1, tid: 2, ts: 0, dur: 20_000 },
        ],
      });
      await writeFile(tracePath, gzipSync(Buffer.from(payload, 'utf8')));
      const summary = await summarizeTrace(tracePath);
      expect(summary.longTasks16ms).toBe(1);
    });
  });

  describe('summary size', () => {
    it('produces a summary under 2000 characters when serialized', () => {
      const summary = computeMetrics(COMPREHENSIVE_EVENTS);
      expect(JSON.stringify(summary).length).toBeLessThan(2000);
    });
  });

  describe('CLI', () => {
    it('outputs JSON when --json flag is present', async () => {
      const tracePath = path.join(tempDir, 'trace.json');
      await writeFile(tracePath, JSON.stringify({ traceEvents: [] }), 'utf8');
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      const scriptPath = path.join(tempDir, 'entry.mjs');
      await runCli([tracePath, '--json'], {
        argv1: scriptPath,
        entryUrl: pathToFileURL(scriptPath).href,
      });
      const expected = JSON.stringify({
        eventCount: 0,
        cpuTimeMs: 0,
        layoutThrashingCount: 0,
        jsExecutionMs: 0,
        paintEventCount: 0,
        memoryPeakBytes: 0,
        droppedFrames: 0,
        longTasks16ms: 0,
        longTasks50ms: 0,
      });
      expect(logSpy).toHaveBeenCalledWith(expected);
      logSpy.mockRestore();
    });

    it('outputs human-readable text without --json flag', async () => {
      const tracePath = path.join(tempDir, 'trace.json');
      await writeFile(tracePath, JSON.stringify({ traceEvents: [] }), 'utf8');
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      const scriptPath = path.join(tempDir, 'entry.mjs');
      await runCli([tracePath], {
        argv1: scriptPath,
        entryUrl: pathToFileURL(scriptPath).href,
      });
      expect(logSpy.mock.calls[0][0]).toContain('Event count:');
      logSpy.mockRestore();
    });

    it('no-ops when not invoked as the entry point', async () => {
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await runCli([], { argv1: undefined, entryUrl: 'file:///nope' });
      expect(logSpy).not.toHaveBeenCalled();
      logSpy.mockRestore();
    });

    it('throws when CLI args lack a trace path', () => {
      expect(() => parseCliArgs([])).toThrow(/Missing trace/);
    });

    it('detects a matching CLI entry point', () => {
      const scriptPath = path.join(tempDir, 'entry.mjs');
      expect(isCliEntryPoint(scriptPath, pathToFileURL(scriptPath).href)).toBe(
        true,
      );
    });

    it('returns false when argv1 is undefined', () => {
      expect(isCliEntryPoint(undefined, 'file:///x')).toBe(false);
    });

    it('returns false when argv1 does not match the entry url', () => {
      const scriptPath = path.join(tempDir, 'entry.mjs');
      const otherPath = path.join(tempDir, 'other.mjs');
      expect(isCliEntryPoint(otherPath, pathToFileURL(scriptPath).href)).toBe(
        false,
      );
    });
  });
});
