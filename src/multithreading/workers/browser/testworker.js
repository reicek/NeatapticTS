import Multi from '../../multi.js';

/** TestWorker Class */
export class TestWorker {
  constructor(dataSet, cost) {
    const blob = new Blob([TestWorker._createBlobString(cost)]);
    this.url = window.URL.createObjectURL(blob);
    this.worker = new Worker(this.url);

    const data = { set: new Float64Array(dataSet).buffer };
    this.worker.postMessage(data, [data.set]);
  }

  evaluate(network) {
    return new Promise((resolve, reject) => {
      const serialized = network.serialize();

      const data = {
        activations: new Float64Array(serialized[0]).buffer,
        states: new Float64Array(serialized[1]).buffer,
        conns: new Float64Array(serialized[2]).buffer,
      };

      this.worker.onmessage = function (e) {
        const error = new Float64Array(e.data.buffer)[0];
        resolve(error);
      };

      this.worker.postMessage(data, [
        data.activations,
        data.states,
        data.conns,
      ]);
    });
  }

  terminate() {
    this.worker.terminate();
    window.URL.revokeObjectURL(this.url);
  }

  static _createBlobString(cost) {
    return `
      const multi = {
        logistic: ${Multi.logistic.toString()},
        tanh: ${Multi.tanh.toString()},
        identity: ${Multi.identity.toString()},
        step: ${Multi.step.toString()},
        relu: ${Multi.relu.toString()},
        softsign: ${Multi.softsign.toString()},
        sinusoid: ${Multi.sinusoid.toString()},
        gaussian: ${Multi.gaussian.toString()},
        bentIdentity: ${Multi.bentIdentity.toString()},
        bipolar: ${Multi.bipolar.toString()},
        bipolarSigmoid: ${Multi.bipolarSigmoid.toString()},
        hardTanh: ${Multi.hardTanh.toString()},
        absolute: ${Multi.absolute.toString()},
        inverse: ${Multi.inverse.toString()},
        selu: ${Multi.selu.toString()},
        deserializeDataSet: ${Multi.deserializeDataSet.toString()},
        testSerializedSet: ${Multi.testSerializedSet.toString()},
        activateSerializedNetwork: ${Multi.activateSerializedNetwork.toString()}
      };

      this.onmessage = function (e) {
        if (typeof e.data.set === 'undefined') {
          const A = new Float64Array(e.data.activations);
          const S = new Float64Array(e.data.states);
          const data = new Float64Array(e.data.conns);

          const error = multi.testSerializedSet(set, cost, A, S, data, multi);

          const answer = { buffer: new Float64Array([error]).buffer };
          postMessage(answer, [answer.buffer]);
        } else {
          set = multi.deserializeDataSet(new Float64Array(e.data.set));
        }
      };`;
  }
}
