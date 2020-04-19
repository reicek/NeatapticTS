import NodeTestWorker from "./node/testworker.js";
import BrowserTestWorker from "./browser/testworker.js";

var workers = {
  node: {
    TestWorker: NodeTestWorker,
  },
  browser: {
    TestWorker: BrowserTestWorker,
  },
};

export default workers;
