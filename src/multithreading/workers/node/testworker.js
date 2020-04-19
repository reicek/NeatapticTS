import cp from "child_process.js";
import path from "path";

/*******************************************************************************
                                WEBWORKER
*******************************************************************************/

function TestWorker(dataSet, cost) {
  const __dirname = path.resolve();

  this.worker = cp.fork(path.join(__dirname, "/worker"));

  this.worker.send({ set: dataSet, cost: cost.name });
}

TestWorker.prototype = {
  evaluate: function (network) {
    return new Promise((resolve) => {
      var serialized = network.serialize();

      var data = {
        activations: serialized[0],
        states: serialized[1],
        conns: serialized[2],
      };

      var _that = this.worker;
      this.worker.on("message", function callback(e) {
        _that.removeListener("message", callback);
        resolve(e);
      });

      this.worker.send(data);
    });
  },

  terminate: function () {
    this.worker.kill();
  },
};

export default TestWorker;
