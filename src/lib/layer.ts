import Neuron from "./neuron";

export enum TActivation {
  ReLU = "ReLU",
  Softmax = "Softmax",
  Sigmoid = "Sigmoid"
}

export default class Layer {
  neurons: Neuron[];
  layerIdx: number;
  learnRate: number;
  activationFunction: TActivation;

  constructor(
    numOfNeurons: number,
    layerIdx: number,
    activationFunction: TActivation,
    numOfWeights: number
  ) {
    this.neurons = new Array(numOfNeurons)
      .fill(0)
      .map(() => new Neuron(layerIdx, activationFunction, numOfWeights));
    this.layerIdx = layerIdx;
    this.learnRate = 0.1;
    this.activationFunction = activationFunction;
  }

  derivativeActivation(value: number) {
    if (this.activationFunction === TActivation.ReLU) {
      // ReLU
      return value > 0 ? 1 : 0;
    }

    if (this.activationFunction === TActivation.Sigmoid) {
      // Sigmoid
      const sVal = 1 / (1 + Math.pow(Math.E, -value));
      return sVal * (1 - sVal);
    }

    if (this.activationFunction === TActivation.Softmax) {
      // Softmax
      const denominator = this.neurons.reduce((prev, curr) => {
        return prev + Math.pow(Math.E, curr.value);
      }, 0);
      const numerator = Math.pow(Math.E, value);
      const result = numerator / denominator;
      return result * (1 - result);
    }
    return 0;
  }

  fire(layers: Layer[]) {
    if (this.layerIdx > 0) {
      this.neurons.forEach((neuron) => neuron.fire(layers));
    }
  }

  backpropagate(sample: { label: number[]; value: number[] }, layers: Layer[]) {
    // Output Layer
    if (this.layerIdx === 2) {
      const hiddenLayer = layers[this.layerIdx - 1];
      this.neurons.forEach((neuron, nIdx) => {
        const dError = -2 * (sample.label[nIdx] - neuron.activeValue);
        const dAct = this.derivativeActivation(neuron.value);
        // Weights
        neuron.weights?.forEach((weight, wIdx) => {
          const dRawInput = hiddenLayer.neurons[wIdx].activeValue;
          const gradient = dError * dAct * dRawInput;
          weight.gradient += gradient;
          // weight.value -= this.learnRate * gradient;
        });

        // Bias
        const gradient = dError * dAct;
        neuron.biasGradient += gradient;
        // neuron.bias -= this.learnRate * gradient;
      });
    }

    // Hidden Layer
    if (this.layerIdx === 1) {
      const inputLayer = layers[this.layerIdx - 1];
      const outputLayer = layers[this.layerIdx + 1];
      this.neurons.forEach((hiddenNeuron, hnIdx) => {
        const dHiddenAct = this.derivativeActivation(hiddenNeuron.value);
        let dError = 0;
        outputLayer.neurons.forEach((outputNeuron, onIdx) => {
          const dOutError =
            -2 * (sample.label[onIdx] - outputNeuron.activeValue);
          const dOutAct = outputLayer.derivativeActivation(outputNeuron.value);
          const dOutRawInput = outputNeuron.weights[hnIdx].value;
          dError += dOutError * dOutAct * dOutRawInput;
        });

        // Weights
        hiddenNeuron.weights?.forEach((weight, wIdx) => {
          const dRawInput = inputLayer.neurons[wIdx].activeValue;
          const gradient = dError * dHiddenAct * dRawInput;
          weight.gradient += gradient;
          // weight.value -= this.learnRate * gradient;
        });

        // Bias
        const gradient = dError * dHiddenAct;
        hiddenNeuron.biasGradient += gradient;
        // hiddenNeuron.bias -= this.learnRate * gradient;
      });
    }
  }

  update(numOfSamples: number) {
    this.neurons.forEach((neuron) => {
      neuron.bias -= (neuron.biasGradient / numOfSamples) * this.learnRate;
      neuron.biasGradient = 0;

      neuron.weights?.forEach((weight) => {
        weight.value -= (weight.gradient / numOfSamples) * this.learnRate;
        weight.gradient = 0;
      });
    });
  }
}
