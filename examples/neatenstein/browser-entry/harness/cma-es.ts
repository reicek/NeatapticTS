/**
 * sep-CMA-ES (separable Covariance Matrix Adaptation Evolution Strategy) for
 * MLP weight optimization in the Neatenstein enemy population.
 *
 * Stub module — implementation lands in Step B4 (04-implementing).
 * Tests in `cma-es.test.ts` define the RED contracts.
 *
 * @module
 */

/**
 * Separable CMA-ES (sep-CMA-ES) for MLP weight optimization.
 *
 * Uses diagonal covariance only, achieving O(n) per-generation complexity
 * instead of the O(n²) of full CMA-ES. This makes it feasible for browser-based
 * 90-weight MLPs. The algorithm maintains a mean vector and a diagonal
 * covariance vector, samples a population, evaluates fitness, and updates the
 * mean and covariance from the top-ranked candidates.
 *
 * @module
 */

/** Default initial step size (standard deviation). */
const DEFAULT_INITIAL_SIGMA = 1.0;

/** Default cumulative step-size adaptation rate. */
const CS = 0.3;

/** Default covariance learning rate (diagonal). */
const CCOV = 0.2;

/** Default learning rate for the mean update. */
const CMEAN = 0.5;

/** Minimal positive floor to prevent covariance collapse. */
const COV_FLOOR = 1e-12;

/** Damping parameter base for step-size adaptation. */
const D_SIGMA_BASE = 1.0;

/**
 * Configuration for creating a sep-CMA-ES state.
 */
export interface SepCmaEsConfig {
  /** Dimensionality of the search space (number of weights). */
  dimension: number;
  /** Initial mean vector (starting point for optimization). */
  initialMean: Float32Array;
  /** Population size (number of candidates per generation). */
  populationSize: number;
  /** Random seed for deterministic sampling. */
  seed: number;
}

/**
 * Internal state of the sep-CMA-ES optimizer.
 */
export interface SepCmaEsState {
  /** Current mean vector. */
  mean: Float32Array;
  /** Diagonal of the covariance vector (variance per dimension). */
  covarianceDiag: Float32Array;
  /** Current step size (standard deviation multiplier). */
  sigma: number;
  /** Population size. */
  populationSize: number;
  /** Dimensionality. */
  dimension: number;
  /** Random seed. */
  seed: number;
  /** Generation counter (incremented each step). */
  generation: number;
  /** Cumulative evolution path for step-size adaptation (CSA). */
  evolutionPath: Float32Array;
}

/**
 * Result of a single sep-CMA-ES evolution step.
 */
export interface SepCmaEsStepResult {
  /** Sampled population of candidate weight vectors. */
  population: Float32Array[];
  /** Updated optimizer state. */
  state: SepCmaEsState;
}

/**
 * Creates a mulberry32 deterministic RNG from a seed.
 *
 * @param seed The random seed.
 * @returns A function returning deterministic floats in [0, 1).
 */
function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generates a standard normal random number N(0, 1) from a uniform RNG using
 * the Box-Muller transform.
 *
 * @param rng Uniform RNG returning [0, 1).
 * @returns Gaussian random number.
 */
function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-10);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Creates a sep-CMA-ES state with diagonal covariance initialized to 1.0.
 *
 * @param config Configuration object with dimension, initialMean, populationSize, and seed.
 * @returns A new CMA-ES state with diagonal covariance.
 */
export function createSepCmaEs(config: SepCmaEsConfig): SepCmaEsState {
  const { dimension, initialMean, populationSize, seed } = config;
  const mean = new Float32Array(dimension);
  mean.set(initialMean);
  const covarianceDiag = new Float32Array(dimension);
  covarianceDiag.fill(DEFAULT_INITIAL_SIGMA);
  const evolutionPath = new Float32Array(dimension);
  return {
    mean,
    covarianceDiag,
    sigma: DEFAULT_INITIAL_SIGMA,
    populationSize,
    dimension,
    seed,
    generation: 0,
    evolutionPath,
  };
}

/**
 * Performs one generation of sep-CMA-ES evolution.
 *
 * Samples a population from N(mean, diag(covariance) * sigma), evaluates
 * fitness, ranks candidates, and updates the mean and diagonal covariance
 * from the top half of the population.
 *
 * @param state The current CMA-ES state.
 * @param fitnessFn Function mapping a weight vector to a fitness score (higher is better).
 * @returns The sampled population and the updated state.
 */
export function stepSepCmaEs(
  state: SepCmaEsState,
  fitnessFn: (weights: Float32Array) => number,
): SepCmaEsStepResult {
  const { dimension, populationSize, mean, covarianceDiag, sigma, seed } = state;
  const rng = createRng(seed + state.generation * 1000003);
  const evolutionPath = state.evolutionPath;

  // 1. Sample population
  const population: Float32Array[] = [];
  for (let p = 0; p < populationSize; p++) {
    const candidate = new Float32Array(dimension);
    for (let d = 0; d < dimension; d++) {
      const std = sigma * Math.sqrt(Math.max(covarianceDiag[d], COV_FLOOR));
      candidate[d] = mean[d] + std * gaussian(rng);
    }
    population.push(candidate);
  }

  // 2. Evaluate and sort by fitness (descending)
  const indexed = population.map((w, i) => ({ w, i, f: fitnessFn(w) }));
  indexed.sort((a, b) => b.f - a.f);

  // 3. Update mean from top half
  const eliteCount = Math.max(1, Math.floor(populationSize / 2));
  const newMean = new Float32Array(dimension);
  let weightSum = 0;
  let weightSqSum = 0;
  for (let k = 0; k < eliteCount; k++) {
    const weight = eliteCount - k;
    weightSum += weight;
    weightSqSum += weight * weight;
  }
  for (let d = 0; d < dimension; d++) {
    let weightedSum = 0;
    for (let k = 0; k < eliteCount; k++) {
      const weight = eliteCount - k;
      weightedSum += indexed[k].w[d] * weight;
    }
    newMean[d] = mean[d] * (1 - CMEAN) + (weightedSum / weightSum) * CMEAN;
  }

  // 4. Update diagonal covariance from top half
  const newCovDiag = new Float32Array(dimension);
  for (let d = 0; d < dimension; d++) {
    let varSum = 0;
    for (let k = 0; k < eliteCount; k++) {
      const diff = indexed[k].w[d] - newMean[d];
      varSum += diff * diff;
    }
    const newVar = varSum / eliteCount;
    newCovDiag[d] =
      covarianceDiag[d] * (1 - CCOV) + newVar * CCOV;
    if (newCovDiag[d] < COV_FLOOR) {
      newCovDiag[d] = COV_FLOOR;
    }
  }

  // 5. Update sigma via Cumulative Step-size Adaptation (CSA)
  //    ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * muEff) * C^(-1/2) * (m' - m) / sigma
  //    sigma *= exp((||ps|| - E) / (dSigma * E))
  const muEff = (weightSum * weightSum) / weightSqSum;
  const psFactor = Math.sqrt(CS * (2 - CS) * muEff);
  const newPs = new Float32Array(dimension);
  for (let d = 0; d < dimension; d++) {
    const meanShift = (newMean[d] - mean[d]) / sigma;
    const invSqrtC = 1 / Math.sqrt(Math.max(covarianceDiag[d], COV_FLOOR));
    newPs[d] = (1 - CS) * evolutionPath[d] + psFactor * invSqrtC * meanShift;
  }
  let psNorm = 0;
  for (let d = 0; d < dimension; d++) {
    psNorm += newPs[d] * newPs[d];
  }
  psNorm = Math.sqrt(psNorm);
  const expected =
    Math.sqrt(dimension) *
    (1 - 1 / (4 * dimension) + 1 / (21 * dimension * dimension));
  const dSigma = D_SIGMA_BASE + CS;
  const newSigma = sigma * Math.exp((psNorm - expected) / (dSigma * expected));

  const newState: SepCmaEsState = {
    mean: newMean,
    covarianceDiag: newCovDiag,
    sigma: newSigma,
    populationSize,
    dimension,
    seed,
    generation: state.generation + 1,
    evolutionPath: newPs,
  };

  return { population, state: newState };
}