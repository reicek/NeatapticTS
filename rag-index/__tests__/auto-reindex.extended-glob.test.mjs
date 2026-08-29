/* global describe, expect, it */

import {
  isReindexableFile,
  reindexChangedFiles,
  resolveStalePlanFixHint,
} from '../auto-reindex.mjs';

describe('auto-reindex extended glob', () => {
  it('identifies reindexable files across corpus families', () => {
    expect(isReindexableFile('plans/foo.plans.md')).toBe(true);
    expect(isReindexableFile('src/neat/network.ts')).toBe(true);
    expect(isReindexableFile('src/neat/network.test.ts')).toBe(false);
    expect(isReindexableFile('src/neat/network.spec.ts')).toBe(false);
    expect(isReindexableFile('src/neat/network.d.ts')).toBe(false);
    expect(isReindexableFile('.github/skills/foo/SKILL.md')).toBe(true);
    expect(isReindexableFile('.github/agents/bar.agent.md')).toBe(true);
    expect(isReindexableFile('.github/copilot-instructions.md')).toBe(true);
    expect(isReindexableFile('README.md')).toBe(false);
    expect(isReindexableFile('rag-index/auto-reindex.mjs')).toBe(false);
  });

  it('reindexes only reindexable changed files', async () => {
    const changedFiles = [
      'src/foo.ts',
      'src/foo.test.ts',
      'src/foo.spec.ts',
      'src/foo.d.ts',
      '.github/skills/foo/SKILL.md',
      '.github/agents/bar.agent.md',
      '.github/copilot-instructions.md',
      'plans/baz.plans.md',
      'package.json',
    ];
    const commands = [];
    const result = await reindexChangedFiles({
      changedFiles,
      runCommand: (command) => {
        commands.push(command);
        return { success: true };
      },
    });

    expect(result.changed).toEqual([
      'src/foo.ts',
      '.github/skills/foo/SKILL.md',
      '.github/agents/bar.agent.md',
      '.github/copilot-instructions.md',
      'plans/baz.plans.md',
    ]);

    expect(commands).toHaveLength(2);
    expect(commands[0][0]).toBe('node');
    expect(commands[0][1]).toBe('rag-index/build-index.mjs');
    expect(commands[1][1]).toBe('rag-index/embed-index.mjs');

    for (const file of result.changed) {
      expect(commands[0]).toContain(`--files=${file}`);
      expect(commands[1]).toContain(`--files=${file}`);
    }

    expect(commands[0]).not.toContain('--files=src/foo.test.ts');
    expect(commands[0]).not.toContain('--files=src/foo.spec.ts');
    expect(commands[0]).not.toContain('--files=src/foo.d.ts');
  });

  it('keeps resolveStalePlanFixHint plan-only', () => {
    const hint = resolveStalePlanFixHint([
      'plans/foo.plans.md',
      'src/foo.ts',
      '.github/skills/foo/SKILL.md',
    ]);

    expect(hint).toContain('--files=plans/foo.plans.md');
    expect(hint).not.toContain('--files=src/foo.ts');
    expect(hint).not.toContain('--files=.github/skills/foo/SKILL.md');
  });
});
