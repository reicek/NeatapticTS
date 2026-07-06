/**
 * Cleanup and primitive-scoping tests for the NGE envelope→Network operator.
 *
 * These tests guard the Phase 3 cleanup contract:
 * - `modeIsEvolvable` is no longer a dead storage field; it is read by the
 *   canonical operator to derive `ngeEnabled`.
 * - Neuromodulation primitives (`ModulatorBroadcaster`, `EpisodicSlot`,
 *   `GatingRouter`) remain descriptor-only and are scoped to a recorded blocker.
 */
import { NGE_DNA } from './neat.nge-dna';
import {
  activateNgeNetworkFromEnvelope,
  NGE_NEUROMODULATION_BLOCKER_ID,
} from './neat.nge-dna.operator';
import type { NetworkJSON } from '../../architecture/network/network.types';
import type { NgeDnaCanonicalEnvelope } from './neat.nge-dna.types';

describe('NGE operator cleanup and primitive scoping', () => {
  it('records a blocker id for descriptor-only neuromodulation primitives', () => {
    expect(NGE_NEUROMODULATION_BLOCKER_ID).toBe('DR-2026-06-27-05-NM');
  });

  it('reads modeIsEvolvable from the envelope to attach the NGE extension carrier', () => {
    // Arrange
    const dna = new NGE_DNA({
      moduleArchetypes: [
        {
          archetypeId: 'archetype:input',
          computationType: 'DenseFeedForward',
        },
        {
          archetypeId: 'archetype:output',
          computationType: 'DenseFeedForward',
        },
      ],
      reproductionPolicy: { modeIsEvolvable: true },
      rulePasses: [
        {
          archetypeId: 'archetype:input',
          kind: 'replicate',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [0.5, 0.5, 0],
            },
          ],
          priority: 0,
        },
        {
          archetypeId: 'archetype:output',
          kind: 'replicate',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [0.5, 0.5, 1],
            },
          ],
          priority: 1,
        },
      ],
    });

    // Act
    const network = activateNgeNetworkFromEnvelope(dna.toCanonical(), 42);
    const json = network.toJSON() as unknown as NetworkJSON;
    const values = json.extensions?.values as Record<
      string,
      NgeDnaCanonicalEnvelope
    >;

    // Assert
    expect(values.ngeEnvelope.reproductionPolicy.modeIsEvolvable).toBe(true);
  });
});
