/**
 * Learning rate methods
 * @see {@link https://stackoverflow.com/questions/30033096/what-is-lr-policy-in-caffe/30045244} 
 */
export default class Rate {
  static fixed() {
    const func = (baseRate, iteration) => {
      return baseRate;
    };

    return func;
  }

  static step(gamma = 0.9, stepSize = 100) {
    const func = (baseRate, iteration) => {
      return baseRate * Math.pow(gamma, Math.floor(iteration / stepSize));
    };

    return func;
  }

  static exp(gamma = 0.999) {
    const func = function (baseRate, iteration) {
      return baseRate * Math.pow(gamma, iteration);
    };

    return func;
  }
  
  static inv (gamma = 0.001, power = 2) {
    const func = function (baseRate, iteration) {
      return baseRate * Math.pow(1 + gamma * iteration, -power);
    };

    return func;
  }
}
