/* global describe, expect, it */

import {
  createFreshnessHook,
  DEFAULT_CHANGED_FILE_GLOBS,
} from '../freshness-hooks/freshness-hooks.mjs';

describe('freshness-hooks extended glob filter', () => {
  it('includes source, script, plan, skill, agent, and copilot-instruction globs', () => {
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('src/**/*.ts');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('scripts/**/*.mjs');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('plans/**/*.md');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('.github/skills/**/*.md');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('.github/agents/**/*.md');
    expect(DEFAULT_CHANGED_FILE_GLOBS).toContain('.github/copilot-instructions.md');
  });

  it('matches the extended corpus families through the watcher globs', async () => {
    const builtPaths = [];
    const hook = createFreshnessHook({
      debounce_ms: 0,
      skip_ann: true,
      client: { execute: async () => ({ rows: [] }) },
      runIncrementalBuild: async (changedPaths) => {
        builtPaths.push(...changedPaths);
        return { updated: changedPaths, failed: [] };
      },
    });

    const changedFiles = [
      'src/foo.ts',
      '.github/skills/foo/SKILL.md',
      '.github/skills/foo/README.md',
      '.github/agents/bar.agent.md',
      '.github/agents/README.md',
      '.github/copilot-instructions.md',
      'plans/baz.plans.md',
      'scripts/qux.mjs',
      'README.md',
      'package.json',
    ];

    for (const filePath of changedFiles) {
      hook.notifyWrite(filePath);
    }

    await hook.flush();

    expect(builtPaths).toEqual([
      'src/foo.ts',
      '.github/skills/foo/SKILL.md',
      '.github/skills/foo/README.md',
      '.github/agents/bar.agent.md',
      '.github/agents/README.md',
      '.github/copilot-instructions.md',
      'plans/baz.plans.md',
      'scripts/qux.mjs',
    ]);
  });
});
