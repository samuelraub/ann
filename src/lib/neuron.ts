import Weight from "./weight";
import Layer, { TActivation } from "./layer";

export default class Neuron {
  layerIdx: number;
  bias: number;
  weights: Weight[];
  value: number;
  activeValue: number;
  biasGradient: number;
  activationFunction: TActivation;

  constructor(
    layerIdx: number,
    activationFunction: TActivation,
    numOfWeights?: number
  ) {
    this.layerIdx = layerIdx;
    this.bias = 0;
    this.weights = new Array(numOfWeights).fill(0).map(() => new Weight());
    this.value = 0;
    this.activeValue = 0;
    this.biasGradient = 0;
    this.activationFunction = activationFunction;
  }

  activation(value: number, neurons: Neuron[]) {
    if (this.activationFunction === TActivation.ReLU) {
      // ReLU
      return value >= 0 ? value : 0;
    }

    if (this.activationFunction === TActivation.Sigmoid) {
      // Sigmoid
      return 1 / (1 + Math.pow(Math.E, -value));
    }

    if (this.activationFunction === TActivation.Softmax) {
      // Softmax
      const denominator = neurons.reduce((prev, curr) => {
        return prev + Math.pow(Math.E, curr.value);
      }, 0);
      const numerator = Math.pow(Math.E, value);
      const result = numerator / denominator;
      return result;
    }
    return 0;
  }

  fire(layers: Layer[]) {
    const prevLayer = layers[this.layerIdx - 1];
    const sum = prevLayer.neurons.reduce((prev, curr, idx) => {
      return prev + curr.activeValue * this.weights[idx].value;
    }, 0);
    const rawValue = sum + this.bias;
    this.value = rawValue;
    this.activeValue = this.activation(rawValue, layers[this.layerIdx].neurons);
  }
}
