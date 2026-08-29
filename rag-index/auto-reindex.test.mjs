import { jest } from '@jest/globals';
import {
  isReindexableFile,
  reindexChangedFiles,
  resolveStalePlanFixHint,
  main,
} from './auto-reindex.mjs';

describe('auto-reindex.mjs', () => {
  describe('reindexChangedFiles', () => {
    it('detects changed plan files and runs reindex commands', async () => {
      const commands = [];
      const result = await reindexChangedFiles({
        changedFiles: ['plans/phase1.plans.md', 'src/neat.test.ts'],
        runCommand: async (cmd) => {
          commands.push(cmd);
          return { success: true };
        },
      });

      expect(result.changed).toEqual(['plans/phase1.plans.md']);
      expect(result.commands).toHaveLength(2);
      expect(result.commands[0]).toContain('--files=plans/phase1.plans.md');
      expect(result.commands[1]).toContain('--files=plans/phase1.plans.md');
      expect(result.failures).toEqual([]);
      expect(result.exitCode).toBe(0);
      expect(result.logPath).toBeTruthy();
      expect(commands).toHaveLength(2);
    });

    it('returns empty commands when no reindexable files changed', async () => {
      const result = await reindexChangedFiles({
        changedFiles: ['src/neat.test.ts', 'README.md'],
        runCommand: async () => ({ success: true }),
      });

      expect(result.changed).toEqual([]);
      expect(result.commands).toEqual([]);
      expect(result.failures).toEqual([]);
      expect(result.exitCode).toBe(0);
    });

    it('records failures when command runner returns success=false', async () => {
      const result = await reindexChangedFiles({
        changedFiles: ['plans/phase1.plans.md'],
        runCommand: async (cmd) => {
          if (cmd[1].includes('embed-index')) return { success: false };
          return { success: true };
        },
      });

      expect(result.failures).toHaveLength(1);
      expect(result.failures[0][1]).toContain('embed-index.mjs');
    });

    it('handles multiple plan files', async () => {
      const result = await reindexChangedFiles({
        changedFiles: ['plans/a.plans.md', 'plans/b.plans.md'],
        runCommand: async () => ({ success: true }),
      });

      expect(result.changed).toEqual(['plans/a.plans.md', 'plans/b.plans.md']);
      expect(result.commands[0]).toContain('--files=plans/a.plans.md');
      expect(result.commands[0]).toContain('--files=plans/b.plans.md');
    });

    it('includes new .github corpus families in reindex commands', async () => {
      const commands = [];
      const result = await reindexChangedFiles({
        changedFiles: [
          '.github/skills/repo-cortex-workflow/SKILL.md',
          '.github/agents/04-implementing.agent.md',
          '.github/copilot-instructions.md',
          'src/foo.test.ts',
        ],
        runCommand: async (cmd) => {
          commands.push(cmd);
          return { success: true };
        },
      });

      expect(result.changed).toEqual([
        '.github/skills/repo-cortex-workflow/SKILL.md',
        '.github/agents/04-implementing.agent.md',
        '.github/copilot-instructions.md',
      ]);
      expect(result.commands).toHaveLength(2);
      expect(result.commands[0]).toContain('--files=.github/skills/repo-cortex-workflow/SKILL.md');
      expect(result.commands[0]).toContain('--files=.github/agents/04-implementing.agent.md');
      expect(result.commands[0]).toContain('--files=.github/copilot-instructions.md');
    });
  });

  describe('resolveStalePlanFixHint', () => {
    it('returns generic hint when no plan files', () => {
      const hint = resolveStalePlanFixHint(['src/neat.ts']);
      expect(hint).toBe(
        'Run: node rag-index/build-index.mjs to rebuild stale index',
      );
    });

    it('returns targeted hint for plan files', () => {
      const hint = resolveStalePlanFixHint(['plans/phase1.plans.md']);
      expect(hint).toContain('--files=plans/phase1.plans.md');
      expect(hint).toContain('build-index.mjs');
      expect(hint).toContain('embed-index.mjs');
    });

    it('filters non-plan files from the hint', () => {
      const hint = resolveStalePlanFixHint([
        'src/neat.ts',
        'plans/phase1.plans.md',
      ]);
      expect(hint).not.toContain('src/neat.ts');
      expect(hint).toContain('plans/phase1.plans.md');
    });

    it('returns generic hint for empty array', () => {
      const hint = resolveStalePlanFixHint([]);
      expect(hint).toBe(
        'Run: node rag-index/build-index.mjs to rebuild stale index',
      );
    });

    it('handles non-string entries gracefully', () => {
      const hint = resolveStalePlanFixHint([null, 123, undefined]);
      expect(hint).toBe(
        'Run: node rag-index/build-index.mjs to rebuild stale index',
      );
    });
  });

  describe('detectChangedFiles via reindexChangedFiles', () => {
    it('uses detectChangedFiles when changedFiles not provided', async () => {
      const result = await reindexChangedFiles({});
      // When git is unavailable, no files are detected. When git is available,
      // only reindexable families are returned.
      expect(result.changed.every((filePath) => isReindexableFile(filePath))).toBe(true);
      expect(result.commands.length).toBe(result.changed.length > 0 ? 2 : 0);
      expect(result.failures).toEqual([]);
      expect(result.exitCode).toBe(0);
      expect(result.logPath).toBeTruthy();
    });
  });

  describe('main', () => {
    it('runs without throwing and logs summary', async () => {
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await main();
      expect(logSpy).toHaveBeenCalled();
      const output = logSpy.mock.calls[0][0];
      expect(output).toContain('"changed"');
      logSpy.mockRestore();
    });
  });
});