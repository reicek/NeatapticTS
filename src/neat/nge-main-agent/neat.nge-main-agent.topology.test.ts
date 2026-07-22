/**
 * Red-phase test contracts for Phase 4 Step 01 — NGE main agent topology budget.
 *
 * Covers AC-402: tier-capped topology is enforced at every lifecycle transition;
 * max nodes/edges respect the configured tier budget.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { NgeMainAgentLifecycleConfig } from './neat.nge-main-agent.types';

import {
  buildMainAgentEmbryo,
  computeEmbryoTopologyBudget,
} from './neat.nge-main-agent.embryo';
import { growJuvenileTopology } from './neat.nge-main-agent.juvenile';
import {
  evaluateAdultTopologyBudget,
  pruneAdultTopology,
} from './neat.nge-main-agent.adult';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 256,
  maxEdges: 1024,
};

describe('NGE main agent topology budget', () => {
  describe('computeEmbryoTopologyBudget', () => {
    it('returns a budget with positive maxNodes and maxEdges', () => {
      // Act
      const budget = computeEmbryoTopologyBudget(defaultConfig);

      // Assert
      expect(budget.maxNodes).toBeGreaterThan(0);
    });

    it('caps embryo nodes at the configured tier maxNodes', () => {
      // Act
      const budget = computeEmbryoTopologyBudget(defaultConfig);

      // Assert
      expect(budget.maxNodes).toBeLessThanOrEqual(defaultConfig.maxNodes);
    });

    it('caps embryo edges at the configured tier maxEdges', () => {
      // Act
      const budget = computeEmbryoTopologyBudget(defaultConfig);

      // Assert
      expect(budget.maxEdges).toBeLessThanOrEqual(defaultConfig.maxEdges);
    });
  });

  describe('buildMainAgentEmbryo', () => {
    it('produces an embryo whose node count is within budget', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.nodeCount).toBeLessThanOrEqual(defaultConfig.maxNodes);
    });

    it('produces an embryo whose edge count is within budget', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.edgeCount).toBeLessThanOrEqual(defaultConfig.maxEdges);
    });
  });

  describe('growJuvenileTopology', () => {
    it('keeps node count within the tier budget', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Act
      const juvenile = growJuvenileTopology(embryo, defaultConfig);

      // Assert
      expect(juvenile.nodeCount).toBeLessThanOrEqual(defaultConfig.maxNodes);
    });

    it('keeps edge count within the tier budget', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Act
      const juvenile = growJuvenileTopology(embryo, defaultConfig);

      // Assert
      expect(juvenile.edgeCount).toBeLessThanOrEqual(defaultConfig.maxEdges);
    });

    it('does not shrink the network below the embryo size', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Act
      const juvenile = growJuvenileTopology(embryo, defaultConfig);

      // Assert
      expect(juvenile.nodeCount).toBeGreaterThanOrEqual(embryo.nodeCount);
    });
  });

  describe('pruneAdultTopology', () => {
    it('keeps node count within the tier budget', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);
      const juvenile = growJuvenileTopology(embryo, defaultConfig);

      // Act
      const adult = pruneAdultTopology(juvenile, defaultConfig);

      // Assert
      expect(adult.nodeCount).toBeLessThanOrEqual(defaultConfig.maxNodes);
    });

    it('keeps edge count within the tier budget', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);
      const juvenile = growJuvenileTopology(embryo, defaultConfig);

      // Act
      const adult = pruneAdultTopology(juvenile, defaultConfig);

      // Assert
      expect(adult.edgeCount).toBeLessThanOrEqual(defaultConfig.maxEdges);
    });
  });

  describe('evaluateAdultTopologyBudget', () => {
    it('reports the adult as within budget when constraints are satisfied', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);
      const juvenile = growJuvenileTopology(embryo, defaultConfig);
      const adult = pruneAdultTopology(juvenile, defaultConfig);

      // Act
      const evaluation = evaluateAdultTopologyBudget(adult, defaultConfig);

      // Assert
      expect(evaluation.withinBudget).toBe(true);
    });
  });
});
