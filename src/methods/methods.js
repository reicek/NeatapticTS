/*******************************************************************************
                                  METHODS
*******************************************************************************/

var methods = {
  activation: import('./activation'),
  mutation: import('./mutation'),
  selection: import('./selection'),
  crossover: import('./crossover'),
  cost: import('./cost'),
  gating: import('./gating'),
  connection: import('./connection'),
  rate: import('./rate'),
};

/** Export */
module.exports = methods;
