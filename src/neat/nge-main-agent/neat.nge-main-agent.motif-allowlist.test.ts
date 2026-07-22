/**
 * Red-phase test contracts for Phase 4 Step 01 — NGE main agent motif allowlist.
 *
 * Covers AC-403: the main agent motif set is exactly the existing catalogue
 * allowlist — AttentionHead, GatedRecurrentCell, EpisodicSlot — with no new
 * computation types or schema version bump.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import {
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
  type NeatGenomeComputationType,
  type NeatGenomeModuleArchetypeDescriptor,
} from '../genome/genome.types';

import type { NgeMainAgentLifecycleConfig } from './neat.nge-main-agent.types';
import {
  buildMainAgentEmbryo,
  resolveMainAgentMotifAllowlist,
} from './neat.nge-main-agent.embryo';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 512,
  maxEdges: 2048,
};

describe('NGE main agent motif allowlist', () => {
  describe('resolveMainAgentMotifAllowlist', () => {
    it('includes AttentionHead in the allowlist', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(allowlist).toContain('AttentionHead');
    });

    it('includes GatedRecurrentCell in the allowlist', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(allowlist).toContain('GatedRecurrentCell');
    });

    it('includes EpisodicSlot in the allowlist', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(allowlist).toContain('EpisodicSlot');
    });

    it('does not include motifs outside the three main-agent motifs', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert — DenseFeedForward is catalogue-valid but not main-agent combat motif
      expect(allowlist).not.toContain('DenseFeedForward');
    });

    it('returns exactly three motifs', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(allowlist).toHaveLength(3);
    });

    it('contains only computation types present in the catalogue', () => {
      // Act
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(
        allowlist.every((motif: NeatGenomeComputationType) =>
          NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE.includes(motif),
        ),
      ).toBe(true);
    });
  });

  describe('buildMainAgentEmbryo', () => {
    it('uses only allowlisted motifs in embryo archetypes', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);
      const allowlist = resolveMainAgentMotifAllowlist();

      // Assert
      expect(
        embryo.archetypes.every(
          (archetype: NeatGenomeModuleArchetypeDescriptor) =>
            allowlist.includes(archetype.computationType),
        ),
      ).toBe(true);
    });

    it('does not invent new computation type strings', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(
        embryo.archetypes.every(
          (archetype: NeatGenomeModuleArchetypeDescriptor) =>
            NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE.includes(
              archetype.computationType,
            ),
        ),
      ).toBe(true);
    });

    it('does not bump the DNA schema version', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.schemaVersion).toBe('A.1.0');
    });
  });
});
