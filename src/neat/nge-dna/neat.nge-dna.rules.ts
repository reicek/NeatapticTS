import { NGE_DNA_SubstrateError } from './neat.nge-dna.errors';
import {
  assignZone,
  buildSubstrateFingerprint,
  buildZoneMap,
  normalizeCoordinate,
} from './neat.nge-dna.substrate';
import { canonicalSerialize, computeFingerprint } from './neat.nge-dna.utils';
import type {
  NgeRulePass,
  NgeSubstrateConfig,
  NgeVirtualModule,
  NgeVirtualModulePlan,
} from './neat.nge-dna.types';

/**
 * Execute one deterministic set of rule passes into an in-memory virtual module plan.
 *
 * @param passes - Rule passes carried by one canonical DNA envelope.
 * @param substrateConfig - Canonical substrate config used for normalization and zoning.
 * @param seed - Deterministic seed folded into the plan fingerprint only.
 * @returns Stable virtual module plan with deterministic ordering and fingerprints.
 * @throws NGE_DNA_SubstrateError When two passes collide on the canonical identity tuple.
 */
export function executeRulePasses(
  passes: NgeRulePass[],
  substrateConfig: NgeSubstrateConfig,
  seed: number,
): NgeVirtualModulePlan {
  const sortedPasses = passes.toSorted(compareRulePasses);

  validateUniqueRulePassIdentities(sortedPasses);

  const substrateFingerprint = buildSubstrateFingerprint(substrateConfig);
  const zoneMap = buildZoneMap(substrateConfig.zonePartition);
  const modules = sortedPasses.flatMap((rulePass, rulePassIndex) =>
    rulePass.placements.map((placement, placementOrdinal) => {
      const coordinate = normalizeCoordinate(placement.coordinate);
      const zoneId = assignZone(coordinate, substrateConfig.zonePartition);
      const zoneDescriptor = zoneMap.get(zoneId) as {
        zoneId: string;
      };

      return {
        moduleId: `${rulePassIndex}:${rulePass.kind}:${rulePass.archetypeId}:${placementOrdinal}`,
        archetypeId: rulePass.archetypeId,
        computationType: placement.computationType,
        coordinate,
        zoneId: zoneDescriptor.zoneId,
        rulePassIndex,
        placementOrdinal,
      } satisfies NgeVirtualModule;
    }),
  );

  return {
    modules,
    substrateFingerprint,
    planFingerprint: computeFingerprint(
      canonicalSerialize({
        modules,
        seed,
        substrateFingerprint,
      }),
    ),
  };
}

function compareRulePasses(
  leftPass: NgeRulePass,
  rightPass: NgeRulePass,
): number {
  return (
    leftPass.priority - rightPass.priority ||
    compareText(leftPass.kind, rightPass.kind) ||
    compareText(leftPass.archetypeId, rightPass.archetypeId)
  );
}

function validateUniqueRulePassIdentities(
  passes: readonly NgeRulePass[],
): void {
  const seenPassIdentityKeys = new Set<string>();

  passes.forEach((rulePass) => {
    const passIdentityKey = `${rulePass.priority}:${rulePass.kind}:${rulePass.archetypeId}`;

    if (seenPassIdentityKeys.has(passIdentityKey)) {
      throw new NGE_DNA_SubstrateError(
        `NGE_DNA rule passes must not repeat the identity tuple ${passIdentityKey}.`,
      );
    }

    seenPassIdentityKeys.add(passIdentityKey);
  });
}

function compareText(leftText: string, rightText: string): number {
  return Number(leftText > rightText) - Number(leftText < rightText);
}
