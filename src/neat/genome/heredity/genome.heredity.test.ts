import Network from '../../../architecture/network';
import { methods } from '../../../neataptic';
import {
  createGenomeFromNetwork,
  selectGenomeHeredityConnectionGenes,
} from '../genome';

function createRandomSequenceGenerator(sequence: number[]): () => number {
  let nextSequenceIndex = 0;

  return () => {
    const resolvedValue =
      sequence.at(nextSequenceIndex) ?? sequence.at(-1) ?? 0;
    nextSequenceIndex += 1;
    return resolvedValue;
  };
}

describe('genome heredity chapter', () => {
  describe('selectGenomeHeredityConnectionGenes', () => {
    describe('given two homologous parent genomes', () => {
      it('preserves the shared innovation set in sorted order', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_411 });
        firstParentNetwork.mutate(methods.mutation.ADD_NODE);
        const secondParentNetwork = firstParentNetwork.clone();
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          equal: true,
          randomGenerator: () => 0,
        });

        // Assert
        expect(
          selectedGenes
            .map((selectedGene) => selectedGene.connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        ).toEqual(
          firstParentGenome.connectionGenes
            .map((connectionGene) => connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        );
      });
    });

    describe('given same-endpoint genes carry different innovations', () => {
      it('keeps the fitter parent innovation lane intact', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_412 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].innovation = 101;
        firstParentNetwork.connections[0].weight = 0.25;
        secondParentNetwork.connections[0].innovation = 202;
        secondParentNetwork.connections[0].weight = 0.75;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 2,
          parent2Score: 1,
          equal: false,
          randomGenerator: () => 0,
        });

        // Assert
        expect({
          innovations: selectedGenes
            .map((selectedGene) => selectedGene.connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          inheritedWeight:
            selectedGenes.find(
              (selectedGene) =>
                selectedGene.connectionGene.innovation === 101,
            )?.connectionGene.weight ?? null,
          hasLowerFitnessInnovation: selectedGenes.some(
            (selectedGene) =>
              selectedGene.connectionGene.innovation === 202,
          ),
        }).toEqual({
          innovations: firstParentGenome.connectionGenes
            .map((connectionGene) => connectionGene.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          inheritedWeight: 0.25,
          hasLowerFitnessInnovation: false,
        });
      });
    });

    describe('given matching disabled genes', () => {
      it('uses the inherited rng for re-enable decisions', () => {
        // Arrange
        const firstParentNetwork = new Network(2, 1, { seed: 2_413 });
        const secondParentNetwork = firstParentNetwork.clone();
        firstParentNetwork.connections[0].enabled = false;
        secondParentNetwork.connections[0].enabled = false;
        const firstParentGenome = createGenomeFromNetwork(firstParentNetwork);
        const secondParentGenome = createGenomeFromNetwork(secondParentNetwork);

        // Act
        const selectedGenes = selectGenomeHeredityConnectionGenes({
          parent1Genome: firstParentGenome,
          parent2Genome: secondParentGenome,
          parent1Score: 1,
          parent2Score: 1,
          parent1ReenableProbability: 0.75,
          parent2ReenableProbability: 0.75,
          equal: false,
          randomGenerator: createRandomSequenceGenerator([0.75, 0.5]),
        });

        // Assert
        expect({
          enabled:
            selectedGenes.find(
              (selectedGene) =>
                selectedGene.connectionGene.innovation ===
                firstParentGenome.connectionGenes[0].innovation,
            )?.connectionGene.enabled ?? null,
        }).toEqual({
          enabled: true,
        });
      });
    });
  });
});