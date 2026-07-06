import { access, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { Buffer } from 'node:buffer';
import { jest } from '@jest/globals';

import {
  parseRepoUrl,
  deriveFolderName,
  assimilateRepo,
} from '../../scripts/assimilation/assimilate-repo.mjs';

describe('assimilate-repo', () => {
  const originalFetch = globalThis.fetch;
  let fetchMock: jest.MockedFunction<(url: string) => Promise<Response>>;
  let tempDir: string | undefined;

  beforeEach(() => {
    fetchMock = jest.fn<(url: string) => Promise<Response>>().mockResolvedValue(
      new Response(JSON.stringify({ tree: [] }), {
        status: 200,
        headers: {
          'content-type': 'application/json',
          'x-ratelimit-remaining': '60',
        },
      }),
    );
    (globalThis as unknown as { fetch: typeof fetchMock }).fetch = fetchMock;
  });

  afterEach(async () => {
    (globalThis as unknown as { fetch: typeof originalFetch }).fetch =
      originalFetch;
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true });
      tempDir = undefined;
    }
  });

  describe('parseRepoUrl', () => {
    it('returns owner, repo and a HEAD ref for a plain GitHub repository URL', () => {
      const result = parseRepoUrl('https://github.com/owner/repo');
      expect(result).toEqual({ owner: 'owner', repo: 'repo', ref: 'HEAD' });
    });

    it('returns owner, repo and the provided branch ref for a tree URL', () => {
      const result = parseRepoUrl('https://github.com/owner/repo/tree/main');
      expect(result).toEqual({ owner: 'owner', repo: 'repo', ref: 'main' });
    });
  });

  describe('deriveFolderName', () => {
    it('defaults to a repo-named subfolder when no outputFolder is provided', () => {
      const result = deriveFolderName('https://github.com/owner/repo');
      expect(path.basename(result.baseFolder)).toBe('repo');
    });

    it('appends the repo name when only a parent outputFolder is provided', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-parent-'));
      const result = deriveFolderName('https://github.com/owner/repo', tempDir);
      expect(result.baseFolder).toBe(path.join(tempDir, 'repo'));
    });
  });

  describe('assimilateRepo', () => {
    it('creates verbatim, notes and repo summary files under the base folder', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-create-'));
      await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      const baseFolder = path.join(tempDir, 'repo');
      const [verbatimExists, notesExists, summaryExists] = await Promise.all([
        access(path.join(baseFolder, 'verbatim'))
          .then(() => true)
          .catch(() => false),
        access(path.join(baseFolder, 'notes'))
          .then(() => true)
          .catch(() => false),
        access(path.join(baseFolder, 'repo.md'))
          .then(() => true)
          .catch(() => false),
      ]);
      expect({ verbatimExists, notesExists, summaryExists }).toEqual({
        verbatimExists: true,
        notesExists: true,
        summaryExists: true,
      });
    });

    it('downloads a text file from the raw URL and preserves its relative path in verbatim', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-raw-'));
      const tree = {
        tree: [
          {
            path: 'src/hello.js',
            mode: '100644',
            type: 'blob',
            sha: 'abc123',
            size: 20,
          },
        ],
      };
      fetchMock.mockImplementation((url: string) => {
        if (url.includes('/git/trees/')) {
          return Promise.resolve(
            new Response(JSON.stringify(tree), {
              status: 200,
              headers: {
                'content-type': 'application/json',
                'x-ratelimit-remaining': '60',
              },
            }),
          );
        }
        if (
          url ===
          'https://raw.githubusercontent.com/owner/repo/HEAD/src/hello.js'
        ) {
          return Promise.resolve(
            new Response("console.log('hello');", { status: 200 }),
          );
        }
        return Promise.resolve(new Response('Not found', { status: 404 }));
      });
      await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      const content = await readFile(
        path.join(tempDir, 'repo', 'verbatim', 'src', 'hello.js'),
        'utf8',
      );
      expect(content).toBe("console.log('hello');");
    });

    it('falls back to the contents API and base64-decodes a file when the raw URL 404s', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-fallback-'));
      const tree = {
        tree: [
          {
            path: 'src/fallback.js',
            mode: '100644',
            type: 'blob',
            sha: 'def456',
            size: 18,
          },
        ],
      };
      const rawUrl =
        'https://raw.githubusercontent.com/owner/repo/HEAD/src/fallback.js';
      const contentsUrl =
        'https://api.github.com/repos/owner/repo/contents/src/fallback.js?ref=HEAD';
      fetchMock.mockImplementation((url: string) => {
        if (url.includes('/git/trees/')) {
          return Promise.resolve(
            new Response(JSON.stringify(tree), {
              status: 200,
              headers: {
                'content-type': 'application/json',
                'x-ratelimit-remaining': '60',
              },
            }),
          );
        }
        if (url === rawUrl) {
          return Promise.resolve(new Response('Not found', { status: 404 }));
        }
        if (url === contentsUrl) {
          const encoded = Buffer.from('fallback content').toString('base64');
          return Promise.resolve(
            new Response(JSON.stringify({ content: encoded }), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            }),
          );
        }
        return Promise.resolve(new Response('Not found', { status: 404 }));
      });
      await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      const content = await readFile(
        path.join(tempDir, 'repo', 'verbatim', 'src', 'fallback.js'),
        'utf8',
      );
      expect(content).toBe('fallback content');
    });

    it('records paths that fail both raw and contents API in notes/fetch-failures.md', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-failures-'));
      const tree = {
        tree: [
          {
            path: 'missing.txt',
            mode: '100644',
            type: 'blob',
            sha: 'ghi789',
            size: 5,
          },
        ],
      };
      fetchMock.mockImplementation((url: string) => {
        if (url.includes('/git/trees/')) {
          return Promise.resolve(
            new Response(JSON.stringify(tree), {
              status: 200,
              headers: {
                'content-type': 'application/json',
                'x-ratelimit-remaining': '60',
              },
            }),
          );
        }
        return Promise.resolve(new Response('Not found', { status: 404 }));
      });
      await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      const failures = await readFile(
        path.join(tempDir, 'repo', 'notes', 'fetch-failures.md'),
        'utf8',
      );
      expect(failures).toContain('missing.txt');
    });

    it('detects a LICENSE file and writes license attribution notes', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-license-'));
      const tree = {
        tree: [
          {
            path: 'LICENSE',
            mode: '100644',
            type: 'blob',
            sha: 'jkl012',
            size: 24,
          },
        ],
      };
      fetchMock.mockImplementation((url: string) => {
        if (url.includes('/git/trees/')) {
          return Promise.resolve(
            new Response(JSON.stringify(tree), {
              status: 200,
              headers: {
                'content-type': 'application/json',
                'x-ratelimit-remaining': '60',
              },
            }),
          );
        }
        if (
          url === 'https://raw.githubusercontent.com/owner/repo/HEAD/LICENSE'
        ) {
          return Promise.resolve(
            new Response('MIT License\nCopyright (c) 2024 Owner', {
              status: 200,
            }),
          );
        }
        return Promise.resolve(new Response('Not found', { status: 404 }));
      });
      await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      const attribution = await readFile(
        path.join(tempDir, 'repo', 'notes', 'license-attribution.md'),
        'utf8',
      );
      expect(attribution).toContain('LICENSE');
    });

    it('exposes the GitHub tree API rate-limit remaining header in the report', async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'assimilate-rate-'));
      fetchMock.mockResolvedValue(
        new Response(JSON.stringify({ tree: [] }), {
          status: 200,
          headers: {
            'content-type': 'application/json',
            'x-ratelimit-remaining': '42',
          },
        }),
      );
      const report = await assimilateRepo('https://github.com/owner/repo', {
        outputFolder: tempDir,
      });
      expect(report.rateLimitRemaining).toBe(42);
    });
  });
});
