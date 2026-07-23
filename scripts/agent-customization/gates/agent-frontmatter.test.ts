import { readFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Red tests for the Phase E pull-to-push migration of agent bodies.
 *
 * Covers:
 * - Tier-1 and specialist agent bodies prefer get_slice_context / search_context
 *   over direct read_file for plan and research files.
 * - read_file is demoted to a documented degraded-Cortex fallback only.
 * - No permanent dual-path "Cortex OR read_file" branch remains.
 */

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const AGENT_04_PATH = path.join(
  REPO_ROOT,
  '.github',
  'agents',
  '04-implementing.agent.md',
);
const AGENT_EXECUTOR_PATH = path.join(
  REPO_ROOT,
  '.github',
  'agents',
  'implementation-executor.agent.md',
);
const AGENT_02_PATH = path.join(
  REPO_ROOT,
  '.github',
  'agents',
  '02-researching.agent.md',
);
const SKILL_RESEARCH_PATH = path.join(
  REPO_ROOT,
  '.github',
  'skills',
  'research-methodology',
  'SKILL.md',
);

let agent04Body: string;
let executorBody: string;
let agent02Body: string;
let researchSkillBody: string;

function getMarkdownBody(content: string): string {
  const match = content.match(/^---\s*\n[\s\S]*?\n---\s*\n([\s\S]*)$/);
  return match ? match[1] : content;
}

beforeAll(async () => {
  agent04Body = getMarkdownBody(await readFile(AGENT_04_PATH, 'utf8'));
  executorBody = getMarkdownBody(await readFile(AGENT_EXECUTOR_PATH, 'utf8'));
  agent02Body = getMarkdownBody(await readFile(AGENT_02_PATH, 'utf8'));
  researchSkillBody = getMarkdownBody(
    await readFile(SKILL_RESEARCH_PATH, 'utf8'),
  );
});

describe('Phase E pull-to-push migration — agent frontmatter contracts', () => {
  describe('04-implementing.agent.md', () => {
    it('D1-red: 04-implementing Default Flow instructs get_slice_context before reading plan files', () => {
      const defaultFlowStart = agent04Body.indexOf('## Default Flow');
      const defaultFlowEnd =
        defaultFlowStart >= 0
          ? agent04Body.indexOf('\n## ', defaultFlowStart + 1)
          : -1;
      const defaultFlowSection =
        defaultFlowStart >= 0
          ? agent04Body.slice(
              defaultFlowStart,
              defaultFlowEnd > 0 ? defaultFlowEnd : undefined,
            )
          : '';
      expect(defaultFlowSection).toContain('get_slice_context');
    });

    it('D1-red: 04-implementing agent body documents read_file as degraded-Cortex fallback only', () => {
      expect(agent04Body).toMatch(/degraded[- ]Cortex fallback/i);
    });

    it('D1-red: 04-implementing agent body does not retain a permanent dual-path Cortex/read_file branch', () => {
      const dualPathPattern =
        /(Cortex|get_slice_context|search_context)[\s\S]{0,40}\bOR\b[\s\S]{0,40}read_file|read_file[\s\S]{0,40}\bOR\b[\s\S]{0,40}(Cortex|get_slice_context|search_context)/i;
      expect(agent04Body).not.toMatch(dualPathPattern);
    });
  });

  describe('implementation-executor.agent.md', () => {
    it('D1-red: implementation-executor agent body references get_slice_context', () => {
      expect(executorBody).toContain('get_slice_context');
    });
  });

  describe('02-researching.agent.md', () => {
    it('AC-D2-RED-001: 02-researching agent body references get_slice_context', () => {
      expect(agent02Body).toContain('get_slice_context');
    });

    it('AC-D2-RED-002: 02-researching agent body documents read_file as degraded-Cortex fallback only', () => {
      expect(agent02Body).toMatch(/degraded[- ]Cortex fallback/i);
    });
  });

  describe('research-methodology SKILL.md', () => {
    it('AC-D2-RED-003: research-methodology skill body references get_slice_context as the primary slice context retrieval mechanism', () => {
      expect(researchSkillBody).toContain('get_slice_context');
    });
  });
});
