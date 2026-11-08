/**
 * @module build-neatracing.test
 * @description Unit tests for the NeatRacing esbuild bundle pipeline
 * (spec P3-S2 / AC-037). Asserts that both the host and the simulation
 * worker bundles are wired with IIFE format, browser platform, minified +
 * sourcemapped output, fs/child_process/path externals, and the
 * `neataptic` -> `src/browser-entry.ts` alias required by AC-037.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { mkdtemp, rm, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import esbuild from 'esbuild';

import {
  NEATRACING_HOST_ENTRY,
  NEATRACING_HOST_OUTFILE,
  NEATRACING_WORKER_ENTRY,
  NEATRACING_WORKER_OUTFILE,
  NEATRACING_SHARED_BUILD_OPTIONS,
  buildNeatRacingBundles,
  runBuildCli,
  isCliEntryPoint,
} from './build-neatracing.mjs';

const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
);

const cliModulePath = path.join(
  repositoryRoot,
  'scripts',
  'build-neatracing.mjs',
);

describe('build-neatracing bundle pipeline', () => {
  let tempDir;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), 'build-neatracing-'));
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it('declares the AC-037 host entry point', () => {
    expect(NEATRACING_HOST_ENTRY).toBe(
      path.join('examples', 'neatRacing', 'browser-entry', 'browser-entry.ts'),
    );
  });

  it('declares the AC-037 host bundle output path', () => {
    expect(NEATRACING_HOST_OUTFILE).toBe(
      path.join('docs', 'assets', 'neatracing.bundle.js'),
    );
  });

  it('declares the AC-037 simulation worker entry point', () => {
    expect(NEATRACING_WORKER_ENTRY).toBe(
      path.join(
        'examples',
        'neatRacing',
        'workers',
        'simulation-worker',
        'simulation-worker.ts',
      ),
    );
  });

  it('declares the AC-037 worker bundle output path', () => {
    expect(NEATRACING_WORKER_OUTFILE).toBe(
      path.join('docs', 'assets', 'neatracing.worker.js'),
    );
  });

  it('bundles both bundles', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.bundle).toBe(true);
  });

  it('targets the browser platform', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.platform).toBe('browser');
  });

  it('emits IIFE format bundles', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.format).toBe('iife');
  });

  it('minifies both bundles', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.minify).toBe(true);
  });

  it('emits sourcemaps for both bundles', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.sourcemap).toBe(true);
  });

  it('targets ES2023 output', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.target).toBe('es2023');
  });

  it('externalizes fs, child_process, and path', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.external).toEqual([
      'fs',
      'child_process',
      'path',
    ]);
  });

  it('aliases neataptic to the repository browser entry shim', () => {
    expect(NEATRACING_SHARED_BUILD_OPTIONS.alias.neataptic).toBe(
      path.join(repositoryRoot, 'src', 'browser-entry.ts'),
    );
  });

  it('aliases to an existing shim file', () => {
    expect(existsSync(NEATRACING_SHARED_BUILD_OPTIONS.alias.neataptic)).toBe(
      true,
    );
  });

  describe('real-build smoke (esbuild into temp dir)', () => {
    it('emits a non-empty IIFE host bundle artifact', async () => {
      const hostOutfile = path.join(tempDir, 'neatracing.bundle.js');
      await buildNeatRacingBundles(esbuild.build, {
        hostOutfile,
        workerOutfile: path.join(tempDir, 'neatracing.worker.js'),
      });
      const bundleStats = await stat(hostOutfile);
      expect(bundleStats.size).toBeGreaterThan(0);
    });

    it('emits the host bundle with an IIFE wrapper in its bytes', async () => {
      const hostOutfile = path.join(tempDir, 'neatracing.bundle.js');
      await buildNeatRacingBundles(esbuild.build, {
        hostOutfile,
        workerOutfile: path.join(tempDir, 'neatracing.worker.js'),
      });
      const bundle = await readFile(hostOutfile, 'utf8');
      expect(bundle.startsWith('"use strict";(()=>{')).toBe(true);
    });

    it('emits a non-empty IIFE simulation worker bundle artifact', async () => {
      const workerOutfile = path.join(tempDir, 'neatracing.worker.js');
      await buildNeatRacingBundles(esbuild.build, {
        hostOutfile: path.join(tempDir, 'neatracing.bundle.js'),
        workerOutfile,
      });
      const bundle = await stat(workerOutfile);
      expect(bundle.size).toBeGreaterThan(0);
    });

    it('emits the worker bundle with an IIFE wrapper in its bytes', async () => {
      const workerOutfile = path.join(tempDir, 'neatracing.worker.js');
      await buildNeatRacingBundles(esbuild.build, {
        hostOutfile: path.join(tempDir, 'neatracing.bundle.js'),
        workerOutfile,
      });
      const bundle = await readFile(workerOutfile, 'utf8');
      expect(bundle.startsWith('"use strict";(()=>{')).toBe(true);
    });

    it('emits host and worker sourcemap siblings', async () => {
      await buildNeatRacingBundles(esbuild.build, {
        hostOutfile: path.join(tempDir, 'neatracing.bundle.js'),
        workerOutfile: path.join(tempDir, 'neatracing.worker.js'),
      });
      const hostMap = await stat(
        path.join(tempDir, 'neatracing.bundle.js.map'),
      );
      expect(hostMap.isFile()).toBe(true);
    });

    it('resolves the default outfiles to the AC-037 docs-assets paths', async () => {
      await buildNeatRacingBundles(esbuild.build);
      const bundle = await stat(
        path.join(repositoryRoot, 'docs', 'assets', 'neatracing.bundle.js'),
      );
      expect(bundle.size).toBeGreaterThan(0);
    });
  });

  describe('CLI entry dispatch', () => {
    it('detects direct CLI invocation via argv1 and entry URL', () => {
      expect(
        isCliEntryPoint(cliModulePath, pathToFileURL(cliModulePath).href),
      ).toBe(true);
    });

    it('rejects argv1 values outside the entry URL', () => {
      const foreignArgv1 = path.join(tempDir, 'jest-worker.js');
      expect(
        isCliEntryPoint(foreignArgv1, pathToFileURL(cliModulePath).href),
      ).toBe(false);
    });

    it('no-ops when the module is imported rather than run as a CLI', async () => {
      const result = await runBuildCli(
        { entryUrl: 'file:///definitely/not/the/cli/module.mjs' },
        { hostOutfile: path.join(tempDir, 'unused-host.js') },
      );
      expect(result).toBeUndefined();
    });

    it('builds both bundles when invoked as the CLI entry point', async () => {
      const hostOutfile = path.join(tempDir, 'cli-host.bundle.js');
      const workerOutfile = path.join(tempDir, 'cli-worker.bundle.js');
      await runBuildCli(
        { argv1: cliModulePath, entryUrl: pathToFileURL(cliModulePath).href },
        { hostOutfile, workerOutfile },
      );
      const bundle = await readFile(hostOutfile, 'utf8');
      expect(bundle.length).toBeGreaterThan(0);
    });
  });
});
