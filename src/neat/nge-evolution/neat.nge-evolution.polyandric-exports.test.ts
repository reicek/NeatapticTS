import type { NgeDnaCanonicalEnvelope } from '../nge-dna/neat.nge-dna.types';

import type {
  NgePolyandricInput,
  NgePolyandricDroneInput,
} from './neat.nge-evolution.reproduction';

import type {
  NgePolyandricInput as FacadeNgePolyandricInput,
  NgePolyandricDroneInput as FacadeNgePolyandricDroneInput,
} from './neat.nge-evolution';

describe('polyandric type exports (P2)', () => {
  it('NgePolyandricInput is importable from reproduction.ts', () => {
    const input: NgePolyandricInput = {
      drones: [],
      ngeEnabled: true,
      queen: {} as NgeDnaCanonicalEnvelope,
      queenId: 'queen:test',
    };
    expect(input.queenId).toBe('queen:test');
  });

  it('NgePolyandricDroneInput is importable from reproduction.ts', () => {
    const drone: NgePolyandricDroneInput = {
      dna: {} as NgeDnaCanonicalEnvelope,
      parentId: 'drone:test',
    };
    expect(drone.parentId).toBe('drone:test');
  });

  it('NgePolyandricInput is importable from the facade', () => {
    const input: FacadeNgePolyandricInput = {
      drones: [],
      ngeEnabled: true,
      queen: {} as NgeDnaCanonicalEnvelope,
      queenId: 'queen:facade',
    };
    expect(input.queenId).toBe('queen:facade');
  });

  it('NgePolyandricDroneInput is importable from the facade', () => {
    const drone: FacadeNgePolyandricDroneInput = {
      dna: {} as NgeDnaCanonicalEnvelope,
      parentId: 'drone:facade',
    };
    expect(drone.parentId).toBe('drone:facade');
  });
});
