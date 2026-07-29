/**
 * @module slice_context_archive
 * @description Slice-context retention for completed/compressed plan steps.
 *
 * When `get_slice_context` cannot find a slice in the active `.plans.md` file,
 * it falls back to this module, which reads the companion `.logs.md` archive
 * (derived by replacing the `.plans.md` suffix with `.logs.md`), scans the
 * archived YAML step/slice packets, and rebuilds a slice descriptor that is
 * compatible with the existing context-window pipeline.
 *
 * The returned descriptor carries an `archived: true` marker so callers can tell
 * the context was recovered from the compressed log archive rather than the
 * active plan.
 */
import { readFile } from 'node:fs/promises';

import {
  normalizeTestContracts,
  parsePlanYamlBlock,
} from '../customization-utils.mjs';

/** Markdown fence pattern for YAML blocks. */
const YAML_FENCE_PATTERN = /^```ya?ml\s*$/gmu;

/**
 * Derive the companion `.logs.md` path from a `.plans.md` path.
 *
 * @param {string} planPath - Repo-relative plan path ending in `.plans.md`.
 * @returns {string} Companion log path ending in `.logs.md`.
 */
export function deriveLogsPath(planPath) {
  return planPath.replace(/\.plans\.md$/u, '.logs.md');
}

/**
 * Find a slice descriptor in the archived `.logs.md` companion file.
 *
 * Scans all YAML blocks in the log file for a step packet whose `slices` list
 * contains the requested `slice_id`. If found, builds a descriptor shaped like
 * `buildDescriptorFromSlice` in `neataptic-workflow-mcp.mjs` and marks it with
 * `archived: true`.
 *
 * @param {string} planPath - Repo-relative path to the active `.plans.md` file.
 * @param {string} sliceId - Exact slice identifier to recover.
 * @param {{ readFile?: (path: string, encoding: string) => Promise<string> }} [deps={}] - Optional dependency injection for tests.
 * @returns {Promise<Record<string, unknown> | null>} Archived descriptor, or `null` when no archive or slice is found.
 */
export async function findArchivedSliceDescriptor(
  planPath,
  sliceId,
  deps = {},
) {
  const logsPath = deriveLogsPath(planPath);
  const read = deps.readFile || defaultReadFile;

  let contents;
  try {
    contents = await read(logsPath, 'utf8');
  } catch {
    return null;
  }

  const blocks = extractYamlBlocks(contents);
  for (const yamlText of blocks) {
    const parsed = parsePlanYamlBlock(yamlText);
    const slices = Array.isArray(parsed.slices) ? parsed.slices : [];
    const slice = slices.find((s) => s && s.slice_id === sliceId);
    if (slice) {
      return buildArchivedDescriptor(parsed, slice, sliceId);
    }
  }

  return null;
}

/**
 * Default filesystem reader.
 *
 * @param {string} filePath - Path to read.
 * @param {string} encoding - Character encoding.
 * @returns {Promise<string>} File contents.
 */
async function defaultReadFile(filePath, encoding) {
  return readFile(filePath, encoding);
}

/**
 * Extract every fenced YAML block from Markdown contents.
 *
 * @param {string} contents - Raw Markdown text.
 * @returns {Array<string>} Array of YAML block bodies.
 */
function extractYamlBlocks(contents) {
  const blocks = [];
  let match;

  while ((match = YAML_FENCE_PATTERN.exec(contents)) !== null) {
    const start = match.index + match[0].length;
    const end = contents.indexOf('```', start);
    if (end === -1) {
      continue;
    }
    blocks.push(contents.slice(start, end).trim());
  }

  return blocks;
}

/**
 * Build a slice descriptor from an archived step/slice packet.
 *
 * Mirrors the field layout produced by `buildDescriptorFromSlice` so the same
 * `buildCompactSliceResponse` pipeline can consume it.
 *
 * @param {Record<string, unknown>} stepPacket - Parsed step packet from the archive.
 * @param {Record<string, unknown>} slice - Matching slice object from the packet.
 * @param {string} sliceId - Exact slice identifier requested.
 * @returns {Record<string, unknown>} Descriptor with `archived: true`.
 */
function buildArchivedDescriptor(stepPacket, slice, sliceId) {
  const stepNumber = String(stepPacket.step ?? stepPacket.step_number ?? '');
  const stepTitle = String(stepPacket.title ?? '');
  const stepStatus = String(stepPacket.status ?? slice.status ?? 'DONE');
  const phase = stepPacket.phase ?? stepPacket.phase_number ?? null;
  const allSlices = stepPacket.slices;

  return {
    archived: true,
    slice_id: String(sliceId),
    stepPacket,
    stepNumber,
    stepMetadata: {
      title: stepTitle,
      status: stepStatus,
      tdd_sequence: slice.tdd_sequence ?? stepPacket.tdd_sequence ?? null,
      mode: slice.mode ?? stepPacket.mode ?? null,
      skills: Array.isArray(slice.skills)
        ? slice.skills
        : Array.isArray(stepPacket.skills)
          ? stepPacket.skills
          : [],
      validation: Array.isArray(slice.validation)
        ? slice.validation
        : Array.isArray(stepPacket.validation)
          ? stepPacket.validation
          : [],
      next_step: stepPacket.next_step ?? null,
    },
    sliceTitle: String(slice.title ?? sliceId),
    boundaryNotes: {
      slice_id: String(sliceId),
      title: String(slice.title ?? sliceId),
      status: String(slice.status ?? 'DONE'),
      goal: slice.goal != null ? String(slice.goal) : null,
      phase,
      phase_status: stepPacket.phase_status ?? 'DONE',
      phase_title: stepPacket.phase_title ?? null,
      step: stepNumber,
      step_status: stepStatus,
      estimate_hours: slice.estimate_hours ?? null,
      parallelizable: slice.parallelizable ?? null,
      files_to_change: Array.isArray(slice.files_to_change)
        ? slice.files_to_change
        : [],
      dependencies: Array.isArray(slice.dependencies) ? slice.dependencies : [],
      next_slice: slice.next_slice ?? null,
      slice_history: allSlices.map((s) => ({
        slice_id: String(s.slice_id ?? ''),
        status: String(s.status ?? ''),
        title: String(s.title ?? ''),
      })),
    },
    testContracts: normalizeTestContracts(slice.acceptance_criteria),
  };
}
