import Multi from '../../../multi.js';
import * as methods from '../../../methods/methods.js';

let set = [];
let cost;

process.on('message', (e) => {
  if (typeof e.set === 'undefined') {
    const { activations: A, states: S, conns: data } = e;

    const result = Multi.testSerializedSet(set, cost, A, S, data, Multi);

    process.send(result);
  } else {
    cost = methods.Cost[e.cost];
    set = Multi.deserializeDataSet(e.set);
  }
});
