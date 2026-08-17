import { describe, it, expect } from '@jest/globals';
import { writeFile, unlink, mkdir } from 'node:fs/promises';
import path from 'node:path';

import {
  deriveLogsPath,
  findArchivedSliceDescriptor,
} from './slice-context-archive.mjs';

const REPO_ROOT = process.cwd();
const TEMP_DIR = path.join(REPO_ROOT, 'plans');

async function writeTempFile(relativePath, content) {
  const fullPath = path.join(REPO_ROOT, relativePath);
  await mkdir(path.dirname(fullPath), { recursive: true });
  await writeFile(fullPath, content, 'utf8');
  return fullPath;
}

async function removeTempFile(relativePath) {
  try {
    await unlink(path.join(REPO_ROOT, relativePath));
  } catch {
    // ignore
  }
}

const FULL_YAML = `# Archived Plan

\`\`\`yaml
phase: 2
step: 3
status: '[DONE]'
phase_status: '[DONE]'
phase_title: 'Phase 2 — Implementation'
next_step: 'Step 04'
tdd_sequence: 'red-green-refactor'
mode: 'strict'
skills:
  - skill-a
  - skill-b
validation:
  - 'npm test'
  - 'npm run lint'
slices:
  - slice_id: archived-slice
    title: 'Archived slice'
    status: '[DONE]'
    goal: 'Implement archived feature'
    estimate_hours: 4
    parallelizable: true
    files_to_change:
      - src/feature.ts
      - src/utils.ts
    dependencies:
      - dep-slice-1
    next_slice: 'next-slice'
    acceptance_criteria:
      - id: AC-001
        text: 'Feature works'
        validation: 'npm test'
  - slice_id: other-archived
    title: 'Other archived slice'
    status: '[WIP]'
\`\`\`
`;

const SPARSE_YAML = `# Sparse Archived Plan

\`\`\`yml
step_number: 5
phase_number: 1
status: '[DONE]'
phase_status: '[WIP]'
phase_title: 'Sparse Phase'
slices:
  - slice_id: sparse-slice
    title: 'Sparse slice'
\`\`\`
`;

const NO_MATCH_YAML = `# No match

\`\`\`yaml
step: 1
slices:
  - slice_id: some-other-slice
    title: 'Other'
\`\`\`
`;

const UNCLOSED_FENCE = `# Unclosed

\`\`\`yaml
step: 1
slices:
  - slice_id: unclosed-slice
`;

const MULTI_BLOCK_YAML = `# Multi block

\`\`\`yaml
step: 1
slices:
  - slice_id: first-slice
    title: 'First'
\`\`\`

Some text here.

\`\`\`yaml
step: 2
slices:
  - slice_id: second-slice
    title: 'Second'
    status: '[DONE]'
    goal: 'Second goal'
\`\`\`
`;

const EMPTY_CONTENT = `# No YAML blocks here

Just regular markdown text.
`;

const SKILLS_VALIDATION_FALLBACK = `# Fallback test

\`\`\`yaml
step: 1
status: '[WIP]'
skills:
  - step-skill
validation:
  - 'npm run build'
slices:
  - slice_id: fallback-slice
    title: 'Fallback slice'
    skills: 'not-an-array'
    validation: 'also-not-an-array'
\`\`\`
`;

describe('slice-context-archive', () => {
  describe('deriveLogsPath', () => {
    it('converts .plans.md to .logs.md', () => {
      expect(deriveLogsPath('plans/my-plan.plans.md')).toBe(
        'plans/my-plan.logs.md',
      );
    });

    it('returns path unchanged when no .plans.md suffix', () => {
      expect(deriveLogsPath('plans/my-plan')).toBe('plans/my-plan');
    });
  });

  describe('findArchivedSliceDescriptor', () => {
    it('finds a fully-specified archived slice', async () => {
      const logsPath = 'plans/test-archive-full.logs.md';
      await writeTempFile(logsPath, FULL_YAML);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'archived-slice',
        );
        expect(result).not.toBeNull();
        expect(result.archived).toBe(true);
        expect(result.stepNumber).toBe('3');
        expect(result.sliceTitle).toBe('Archived slice');
        expect(result.boundaryNotes.phase).toBe(2);
        expect(result.boundaryNotes.phase_status).toBe('[DONE]');
        expect(result.boundaryNotes.phase_title).toBe('Phase 2 — Implementation');
        expect(result.boundaryNotes.step).toBe('3');
        expect(result.boundaryNotes.status).toBe('[DONE]');
        expect(result.boundaryNotes.goal).toBe('Implement archived feature');
        expect(result.boundaryNotes.estimate_hours).toBe(4);
        expect(result.boundaryNotes.parallelizable).toBe(true);
        expect(result.boundaryNotes.files_to_change).toEqual([
          'src/feature.ts',
          'src/utils.ts',
        ]);
        expect(result.boundaryNotes.dependencies).toEqual(['dep-slice-1']);
        expect(result.boundaryNotes.next_slice).toBe('next-slice');
        expect(result.stepMetadata.skills).toEqual(['skill-a', 'skill-b']);
        expect(result.stepMetadata.validation).toEqual([
          'npm test',
          'npm run lint',
        ]);
        expect(result.boundaryNotes.slice_history).toHaveLength(2);
        expect(result.boundaryNotes.slice_history[0].slice_id).toBe(
          'archived-slice',
        );
        expect(result.boundaryNotes.slice_history[1].slice_id).toBe(
          'other-archived',
        );
        expect(result.testContracts).toHaveLength(1);
        expect(result.testContracts[0].id).toBe('AC-001');
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('finds a sparse archived slice using fallback paths', async () => {
      const logsPath = 'plans/test-archive-sparse.logs.md';
      await writeTempFile(logsPath, SPARSE_YAML);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'sparse-slice',
        );
        expect(result).not.toBeNull();
        expect(result.archived).toBe(true);
        expect(result.stepNumber).toBe('5');
        expect(result.sliceTitle).toBe('Sparse slice');
        expect(result.boundaryNotes.phase).toBe(1);
        expect(result.boundaryNotes.phase_status).toBe('[WIP]');
        expect(result.boundaryNotes.phase_title).toBe('Sparse Phase');
        expect(result.boundaryNotes.status).toBe('DONE');
        expect(result.boundaryNotes.goal).toBeNull();
        expect(result.boundaryNotes.estimate_hours).toBeNull();
        expect(result.boundaryNotes.parallelizable).toBeNull();
        expect(result.boundaryNotes.files_to_change).toEqual([]);
        expect(result.boundaryNotes.dependencies).toEqual([]);
        expect(result.boundaryNotes.next_slice).toBeNull();
        expect(result.stepMetadata.skills).toEqual([]);
        expect(result.stepMetadata.validation).toEqual([]);
        expect(result.boundaryNotes.slice_history).toHaveLength(1);
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('returns null when no matching slice is found', async () => {
      const logsPath = 'plans/test-archive-no-match.logs.md';
      await writeTempFile(logsPath, NO_MATCH_YAML);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'nonexistent-slice',
        );
        expect(result).toBeNull();
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('returns null for empty content with no YAML blocks', async () => {
      const logsPath = 'plans/test-archive-empty.logs.md';
      await writeTempFile(logsPath, EMPTY_CONTENT);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'any-slice',
        );
        expect(result).toBeNull();
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('handles unclosed YAML fences', async () => {
      const logsPath = 'plans/test-archive-unclosed.logs.md';
      await writeTempFile(logsPath, UNCLOSED_FENCE);
      try {
        // Should not find the slice because the fence is unclosed
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'unclosed-slice',
        );
        expect(result).toBeNull();
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('finds slice in second YAML block', async () => {
      const logsPath = 'plans/test-archive-multi.logs.md';
      await writeTempFile(logsPath, MULTI_BLOCK_YAML);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'second-slice',
        );
        expect(result).not.toBeNull();
        expect(result.sliceTitle).toBe('Second');
        expect(result.boundaryNotes.goal).toBe('Second goal');
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('returns null when readFile throws', async () => {
      const result = await findArchivedSliceDescriptor(
        'plans/nonexistent-file.logs.md',
        'any-slice',
      );
      expect(result).toBeNull();
    });

    it('uses stepPacket skills/validation when slice fields are not arrays', async () => {
      const logsPath = 'plans/test-archive-fallback.logs.md';
      await writeTempFile(logsPath, SKILLS_VALIDATION_FALLBACK);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'fallback-slice',
        );
        expect(result).not.toBeNull();
        // skills should fall back to stepPacket.skills since slice.skills is a string
        expect(result.stepMetadata.skills).toEqual(['step-skill']);
        // validation should fall back to stepPacket.validation
        expect(result.stepMetadata.validation).toEqual(['npm run build']);
      } finally {
        await removeTempFile(logsPath);
      }
    });

    it('uses defaultReadFile when deps.readFile is not provided', async () => {
      const logsPath = 'plans/test-archive-default-read.logs.md';
      const content = `\`\`\`yaml
step: 1
slices:
  - slice_id: default-read-slice
    title: 'Default read test'
\`\`\`
`;
      await writeTempFile(logsPath, content);
      try {
        const result = await findArchivedSliceDescriptor(
          logsPath,
          'default-read-slice',
        );
        expect(result).not.toBeNull();
        expect(result.sliceTitle).toBe('Default read test');
      } finally {
        await removeTempFile(logsPath);
      }
    });
  });
});