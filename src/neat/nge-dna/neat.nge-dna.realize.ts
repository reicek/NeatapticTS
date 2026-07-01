import {
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
  type NeatGenomeComputationType,
  type NeatGenomeSubstrateCoordinate,
} from '../genome/genome.types';

import { evaluateCppnProgram } from './neat.nge-dna.cppn';
import { NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD } from './neat.nge-dna.constants';
import { NGE_DNA_CppnError } from './neat.nge-dna.errors';
import { canonicalSerialize, computeFingerprint } from './neat.nge-dna.utils';
import type {
  NgeDnaCanonicalEnvelope,
  NgeDnaModuleArchetype,
  NgeRealizedEdge,
  NgeRealizedModule,
  NgeRealizedPhenotypeDescriptor,
  NgeVirtualModule,
  NgeVirtualModulePlan,
} from './neat.nge-dna.types';

const KNOWN_COMPUTATION_TYPES = new Set<NeatGenomeComputationType>(
  NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE,
);

/**
 * Materialize one serializable phenotype descriptor from the deterministic virtual module plan.
 *
 * @param plan - Canonical virtual module plan emitted by substrate rule execution.
 * @param envelope - Canonical DNA envelope carrying CPPN programs and archetype metadata.
 * @param seed - Deterministic seed folded into the realized phenotype fingerprint.
 * @returns Fully JSON-serializable realized phenotype descriptor.
 * @throws NGE_DNA_CppnError When one virtual-module computation type is unknown.
 */
export function realizePhenotypeFromPlan(
  plan: NgeVirtualModulePlan,
  envelope: NgeDnaCanonicalEnvelope,
  seed: number,
): NgeRealizedPhenotypeDescriptor {
  // Step 1: Resolve the archetype lookup used to decorate the virtual modules.
  const archetypeById = new Map<string, NgeDnaModuleArchetype>(
    envelope.moduleArchetypes.map((moduleArchetype) => [
      moduleArchetype.archetypeId,
      moduleArchetype,
    ]),
  );

  // Step 2: Validate the public computation-type dispatch contract.
  validateComputationTypes(plan.modules);

  // Step 3: Build the realized modules and the archetype-derived assignment shelves.
  const realizedModules = plan.modules.map((virtualModule) =>
    realizeModule(virtualModule, archetypeById.get(virtualModule.archetypeId)),
  );
  const residualStreamAssignments = collectAssignmentShelf(
    realizedModules,
    'residualStreamId',
  );
  const weightSharedCohortAssignments = collectAssignmentShelf(
    realizedModules,
    'weightSharedCohortId',
  );

  // Step 4: Realize directed adjacency with cost-aware thresholding.
  const edges =
    envelope.cppnPrograms.length === 0
      ? []
      : realizeDirectedEdges(realizedModules, envelope.cppnPrograms[0]);

  // Step 5: Fold the canonical realized content into the final descriptor.
  return {
    modules: realizedModules,
    edges,
    residualStreamAssignments,
    weightSharedCohortAssignments,
    phenotypeFingerprint: computeFingerprint(
      canonicalSerialize({
        edges,
        modules: realizedModules,
        seed,
      }),
    ),
    dnaFingerprint: envelope.fingerprint,
    seed,
  };
}

function validateComputationTypes(modules: readonly NgeVirtualModule[]): void {
  modules.forEach((virtualModule) => {
    if (!KNOWN_COMPUTATION_TYPES.has(virtualModule.computationType)) {
      throw new NGE_DNA_CppnError(
        `NGE_DNA virtual module ${virtualModule.moduleId} uses unknown computationType ${virtualModule.computationType}.`,
      );
    }
  });
}

function realizeModule(
  virtualModule: NgeVirtualModule,
  moduleArchetype: NgeDnaModuleArchetype | undefined,
): NgeRealizedModule {
  return {
    moduleId: virtualModule.moduleId,
    archetypeId: virtualModule.archetypeId,
    computationType: virtualModule.computationType,
    coordinate: cloneCoordinate(virtualModule.coordinate),
    zoneId: virtualModule.zoneId,
    receivesCoordinates: moduleArchetype?.receivesCoordinates ?? false,
    residualStreamId: moduleArchetype?.residualStreamId,
    weightSharedCohortId: moduleArchetype?.weightSharedCohortId,
    archetypeParams:
      moduleArchetype?.parameterSchema === undefined
        ? undefined
        : structuredClone(moduleArchetype.parameterSchema),
  };
}

function collectAssignmentShelf(
  modules: readonly NgeRealizedModule[],
  assignmentKey: 'residualStreamId' | 'weightSharedCohortId',
): Record<string, string[]> {
  return modules.reduce<Record<string, string[]>>(
    (assignmentShelf, moduleDescriptor) => {
      const assignmentId = moduleDescriptor[assignmentKey];

      if (assignmentId !== undefined) {
        assignmentShelf[assignmentId] = [
          ...(assignmentShelf[assignmentId] ?? []),
          moduleDescriptor.moduleId,
        ];
      }

      return assignmentShelf;
    },
    {},
  );
}

function realizeDirectedEdges(
  modules: readonly NgeRealizedModule[],
  cppnProgram: NgeDnaCanonicalEnvelope['cppnPrograms'][number],
): NgeRealizedEdge[] {
  return modules.flatMap((sourceModule) =>
    modules.flatMap((targetModule) => {
      if (sourceModule.moduleId === targetModule.moduleId) {
        return [];
      }

      const euclideanDistance = computeEuclideanDistance(
        sourceModule.coordinate,
        targetModule.coordinate,
      );
      const [cppnWeight] = evaluateCppnProgram(cppnProgram, [
        sourceModule.coordinate[0],
        sourceModule.coordinate[1],
        sourceModule.coordinate[2],
        targetModule.coordinate[0],
        targetModule.coordinate[1],
        targetModule.coordinate[2],
        euclideanDistance,
      ]);

      if (Math.abs(cppnWeight) < NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD) {
        return [];
      }

      const broadcastRadius = readBroadcastRadius(sourceModule.archetypeParams);
      const isResidualTap = sourceModule.residualStreamId !== undefined;
      const isModulatorBroadcast =
        sourceModule.computationType === 'ModulatorBroadcaster' &&
        typeof broadcastRadius === 'number' &&
        broadcastRadius >= euclideanDistance;

      return [
        {
          sourceModuleId: sourceModule.moduleId,
          targetModuleId: targetModule.moduleId,
          weight: cppnWeight,
          isResidualTap,
          isModulatorBroadcast,
          wiringCost:
            isResidualTap || isModulatorBroadcast ? 0 : euclideanDistance,
        },
      ];
    }),
  );
}

function readBroadcastRadius(
  archetypeParams: Record<string, unknown> | undefined,
): number | undefined {
  const broadcastRadius = archetypeParams?.broadcastRadius;
  return typeof broadcastRadius === 'number' ? broadcastRadius : undefined;
}

function computeEuclideanDistance(
  sourceCoordinate: NeatGenomeSubstrateCoordinate,
  targetCoordinate: NeatGenomeSubstrateCoordinate,
): number {
  return Math.sqrt(
    sourceCoordinate.reduce(
      (sumOfSquares, sourceAxisCoordinate, axisIndex) =>
        sumOfSquares +
        (sourceAxisCoordinate - targetCoordinate[axisIndex]) ** 2,
      0,
    ),
  );
}

function cloneCoordinate(
  coordinate: NeatGenomeSubstrateCoordinate,
): NeatGenomeSubstrateCoordinate {
  return [coordinate[0], coordinate[1], coordinate[2]];
}
