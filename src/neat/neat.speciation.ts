/**
 * Assign genomes into species based on compatibility distance and maintain species structures.
 * This function creates new species for unassigned genomes, prunes empty species, updates
 * dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
 * records per‑species history statistics used by telemetry and adaptive controllers.
 *
 * Implementation notes:
 */
import type {
  NeatLike,
  GenomeDetailed,
  SpeciesLike,
  ConnectionLike,
  SpeciationOptions,
  SpeciationHarnessContext,
} from './neat.types';

// Compact, typed implementations for species maintenance helpers.

export function _speciate<
  TOptions extends SpeciationOptions = SpeciationOptions,
>(this: SpeciationHarnessContext<TOptions>) {
  this._prevSpeciesMembers = this._prevSpeciesMembers ?? new Map();
  this._prevSpeciesMembers.clear();
  for (const species of this._species) {
    const previousMembers = new Set<number>();
    for (const member of species.members as GenomeDetailed[])
      previousMembers.add(member._id);
    this._prevSpeciesMembers.set(species.id, previousMembers);
  }

  this._species.forEach((species) => (species.members = []));

  for (const genome of this.population) {
    let isAssigned = false;
    for (const species of this._species) {
      const compatibilityDistance = this._compatibilityDistance(
        genome,
        species.representative as GenomeDetailed,
      );
      if (compatibilityDistance < (this.options.compatibilityThreshold ?? 3)) {
        species.members.push(genome);
        isAssigned = true;
        break;
      }
    }
    if (!isAssigned) {
      const newSpeciesId = this._nextSpeciesId++;
      this._species.push({
        id: newSpeciesId,
        members: [genome],
        representative: genome,
        lastImproved: this.generation,
        bestScore: genome.score ?? -Infinity,
      });
      this._speciesCreated.set(newSpeciesId, this.generation);
    }
  }

  // --- PID controller for dynamic compatibility threshold ---
  // Only run if a targetSpecies is set
  const options = this.options;
  // Use compatAdjust if present, else fallback to top-level or defaults
  const compatAdjust = options.compatAdjust ?? {};
  const minThreshold = compatAdjust.minThreshold ?? options.minThreshold ?? 1;
  const maxThreshold = compatAdjust.maxThreshold ?? options.maxThreshold ?? 10;
  const targetSpeciesCount = options.targetSpecies ?? 5;
  const observedSpeciesCount = this._species.length;
  // Ensure the integral is always initialized so PID always runs
  if (typeof this._compatIntegral !== 'number') this._compatIntegral = 0;
  if (typeof options.compatibilityThreshold === 'number') {
    // PID controller: error = target - observed (correct sign)
    const speciesError = targetSpeciesCount - observedSpeciesCount;
    // PID constants (tunable)
    const proportionalGain = compatAdjust.kp ?? 0.5;
    const integralGain = compatAdjust.ki ?? 10;
    // Proportional
    let thresholdDelta = proportionalGain * speciesError;
    // Integral
    this._compatIntegral += speciesError;
    thresholdDelta += integralGain * this._compatIntegral;
    // Update threshold
    let updatedThreshold = options.compatibilityThreshold - thresholdDelta;
    // Strict clamping: set to min or max if out of bounds, and reset integral
    if (updatedThreshold < minThreshold) {
      updatedThreshold = minThreshold;
      this._compatIntegral = 0;
    } else if (updatedThreshold > maxThreshold) {
      updatedThreshold = maxThreshold;
      this._compatIntegral = 0;
    }

    options.compatibilityThreshold = updatedThreshold;
  }

  // Always clamp compatibilityThreshold to min/max after PID controller or assignment
  if (typeof options.compatibilityThreshold === 'number') {
    if (options.compatibilityThreshold < minThreshold)
      options.compatibilityThreshold = minThreshold;
    if (options.compatibilityThreshold > maxThreshold)
      options.compatibilityThreshold = maxThreshold;
  }

  this._species = this._species.filter((species) => species.members.length > 0);
  this._species.forEach(
    (species) =>
      (species.representative = species.members[0] as GenomeDetailed),
  );

  const ageProtection = options.speciesAgeProtection ?? {
    grace: 3,
    oldPenalty: 0.5,
  };
  for (const species of this._species) {
    const createdGeneration =
      this._speciesCreated.get(species.id) ?? this.generation;
    const speciesAge = this.generation - createdGeneration;
    if (speciesAge >= (ageProtection.grace ?? 3) * 10) {
      const penalty = ageProtection.oldPenalty ?? 0.5;
      if (penalty < 1)
        (species.members as GenomeDetailed[]).forEach((member) => {
          if (typeof member.score === 'number') member.score! *= penalty;
        });
    }
  }

  // compact history snapshot (extended when requested)
  if (options.speciesAllocation?.extendedHistory) {
    const stats = this._species.map((species) => {
      const members = species.members as GenomeDetailed[];
      const sizes = members.map((member) => ({
        nodes: member.nodes.length,
        conns: member.connections.length,
        score: member.score ?? 0,
        ent: this._structuralEntropy(member),
      }));
      const average = (arr: number[]) =>
        arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
      const meanNodes = average(sizes.map((x) => x.nodes));
      const meanConns = average(sizes.map((x) => x.conns));
      let innovationSum = 0;
      let innovationCount = 0;
      let maxInnovation = -Infinity;
      let minInnovation = Infinity;
      let enabledCount = 0;
      let disabledCount = 0;
      for (const member of members)
        for (const connection of member.connections as ConnectionLike[]) {
          const innovation =
            connection.innovation ?? this._fallbackInnov(connection);
          innovationSum += innovation;
          innovationCount++;
          if (innovation > maxInnovation) maxInnovation = innovation;
          if (innovation < minInnovation) minInnovation = innovation;
          if (connection.enabled === false) disabledCount++;
          else enabledCount++;
        }
      const meanInnovation = innovationCount
        ? innovationSum / innovationCount
        : 0;
      return {
        id: species.id,
        size: species.members.length,
        best: species.bestScore,
        meanNodes,
        meanConns,
        meanInnovation,
        innovationRange:
          isFinite(maxInnovation) &&
          isFinite(minInnovation) &&
          maxInnovation > minInnovation
            ? maxInnovation - minInnovation
            : 0,
        enabledRatio:
          enabledCount + disabledCount
            ? enabledCount / (enabledCount + disabledCount)
            : 0,
      } as Record<string, unknown>;
    });
    this._speciesHistory.push({ generation: this.generation, stats });
  } else {
    this._speciesHistory.push({
      generation: this.generation,
      stats: this._species.map((species) => ({
        id: species.id,
        size: species.members.length,
        best: species.bestScore,
      })),
    });
  }

  if (this._speciesHistory.length > 200) this._speciesHistory.shift();
}

export function _applyFitnessSharing(
  this: NeatLike & {
    _species: SpeciesLike[];
    _compatibilityDistance: (a: GenomeDetailed, b: GenomeDetailed) => number;
  },
) {
  const sigma = (this.options as any).sharingSigma ?? 0;
  if (sigma > 0) {
    for (const s of this._species) {
      const members = s.members as GenomeDetailed[];
      for (let i = 0; i < members.length; i++) {
        const mi = members[i];
        if (typeof mi.score !== 'number') continue;
        let sum = 0;
        for (let j = 0; j < members.length; j++) {
          const mj = members[j];
          const d = i === j ? 0 : this._compatibilityDistance(mi, mj);
          if (d < sigma) {
            const r = d / sigma;
            sum += 1 - r * r;
          }
        }
        if (sum <= 0) sum = 1;
        mi.score = mi.score / sum;
      }
    }
  } else {
    for (const s of this._species) {
      const members = s.members as GenomeDetailed[];
      const size = members.length || 1;
      for (const m of members)
        if (typeof m.score === 'number') m.score = m.score / size;
    }
  }
}

export function _sortSpeciesMembers(this: NeatLike, sp: SpeciesLike) {
  (sp.members as GenomeDetailed[]).sort(
    (a, b) => (b.score || 0) - (a.score || 0),
  );
}

export function _updateSpeciesStagnation(
  this: NeatLike & { _species: SpeciesLike[]; generation: number },
) {
  const win = (this.options as any).stagnationGenerations ?? 15;
  for (const s of this._species) {
    _sortSpeciesMembers.call(this, s);
    const top = (s.members as GenomeDetailed[])[0];
    if ((top?.score ?? -Infinity) > (s.bestScore ?? -Infinity)) {
      s.bestScore = top.score ?? -Infinity;
      s.lastImproved = this.generation;
    }
  }
  const survivors = this._species.filter(
    (s) => this.generation - (s.lastImproved ?? 0) <= win,
  );
  if (survivors.length) this._species = survivors;
}
