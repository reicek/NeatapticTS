import { DEFAULTS, ROUTING, classifyAndRoute } from './routing-table.mjs';

describe('DEFAULTS', () => {
  it('has alpha for all 6 classes', () => {
    expect(DEFAULTS.simple_lookup).toBe(0.75);
    expect(DEFAULTS.cross_boundary).toBe(0.5);
    expect(DEFAULTS.multi_hop).toBe(0.35);
    expect(DEFAULTS.exploratory).toBe(0.3);
    expect(DEFAULTS.code_specific).toBe(0.7);
    expect(DEFAULTS.plan_specific).toBe(0.7);
  });
});

describe('ROUTING', () => {
  it('has routing for simple_lookup', () => {
    expect(ROUTING.simple_lookup).toEqual({ family: null, expansion: 'none', post_processing: 'default' });
  });

  it('has routing for cross_boundary', () => {
    expect(ROUTING.cross_boundary).toEqual({ family: null, expansion: 'multi_family', post_processing: 'cross_family_dedup' });
  });

  it('has routing for multi_hop', () => {
    expect(ROUTING.multi_hop).toEqual({ family: null, expansion: 'entity_graph', post_processing: 'hop_decay' });
  });

  it('has routing for exploratory', () => {
    expect(ROUTING.exploratory).toEqual({ family: null, expansion: 'context_assembly', post_processing: 'budget_assembly' });
  });

  it('has routing for code_specific', () => {
    expect(ROUTING.code_specific).toEqual({ family: 'ts-source', expansion: 'none', post_processing: 'default' });
  });

  it('has routing for plan_specific', () => {
    expect(ROUTING.plan_specific).toEqual({ family: 'plan,completed-plan', expansion: 'none', post_processing: 'default' });
  });
});

describe('classifyAndRoute', () => {
  it('classifies and routes simple_lookup without hints', () => {
    const result = classifyAndRoute('crossover');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.9);
    expect(result.alpha).toBe(0.75);
    expect(result.strategy).toEqual({ family: null, expansion: 'none', post_processing: 'default' });
  });

  it('classifies and routes code_specific without hints', () => {
    const result = classifyAndRoute('network.activate');
    expect(result.query_class).toBe('code_specific');
    expect(result.alpha).toBe(0.7);
    expect(result.strategy.family).toBe('ts-source');
  });

  it('overrides alpha with classification_hints', () => {
    const result = classifyAndRoute('crossover', { alpha: 0.9 });
    expect(result.alpha).toBe(0.9);
    expect(result.strategy.family).toBeNull();
  });

  it('overrides family with classification_hints', () => {
    const result = classifyAndRoute('crossover', { family: 'ts-source' });
    expect(result.strategy.family).toBe('ts-source');
  });

  it('overrides both alpha and family with classification_hints', () => {
    const result = classifyAndRoute('network.activate', { alpha: 0.5, family: 'plan' });
    expect(result.alpha).toBe(0.5);
    expect(result.strategy.family).toBe('plan');
  });

  it('handles null classification_hints', () => {
    const result = classifyAndRoute('crossover', null);
    expect(result.alpha).toBe(0.75);
    expect(result.strategy.family).toBeNull();
  });

  it('handles undefined classification_hints', () => {
    const result = classifyAndRoute('crossover');
    expect(result.alpha).toBe(0.75);
    expect(result.strategy.family).toBeNull();
  });

  it('handles classification_hints with family set to null explicitly', () => {
    const result = classifyAndRoute('network.activate', { family: null });
    expect(result.strategy.family).toBeNull();
  });

  it('handles classification_hints with alpha but no family', () => {
    const result = classifyAndRoute('network.activate', { alpha: 0.3 });
    expect(result.alpha).toBe(0.3);
    expect(result.strategy.family).toBe('ts-source');
  });

  it('routes plan_specific', () => {
    const result = classifyAndRoute('what is the checkpointing design and architecture');
    expect(result.query_class).toBe('plan_specific');
    expect(result.alpha).toBe(0.7);
    expect(result.strategy.family).toBe('plan,completed-plan');
  });

  it('routes exploratory', () => {
    const result = classifyAndRoute('how does the training pipeline work with data');
    expect(result.query_class).toBe('exploratory');
    expect(result.alpha).toBe(0.3);
  });

  it('routes multi_hop', () => {
    const result = classifyAndRoute('functions that call activate and also use slab in network');
    expect(result.query_class).toBe('multi_hop');
    expect(result.alpha).toBe(0.35);
  });

  it('routes cross_boundary', () => {
    const result = classifyAndRoute('the relationship between crossover and mutation in system');
    expect(result.query_class).toBe('cross_boundary');
    expect(result.alpha).toBe(0.5);
  });

  it('routes fallback simple_lookup', () => {
    const result = classifyAndRoute('find all the things in the system that are not related to anything else here');
    expect(result.query_class).toBe('simple_lookup');
    expect(result.confidence).toBe(0.5);
    expect(result.alpha).toBe(0.75);
  });
});