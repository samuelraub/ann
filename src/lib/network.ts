import Layer, { TActivation } from "./layer";

export type TTrainingData = {
  value: number[];
  label: number[];
}[];

export default class Network {
  layers: Layer[];
  outputValues: number[][];
  mse: number[];

  constructor(layers: [number, number, number]) {
    this.layers = layers.map((numOfNodes, idx) => {
      if (idx === 0) {
        return new Layer(numOfNodes, idx, TActivation.ReLU, 0);
      }
      if (idx === 1) {
        return new Layer(numOfNodes, idx, TActivation.ReLU, layers[0]);
      }
      if (idx === 2) {
        return new Layer(numOfNodes, idx, TActivation.Sigmoid, layers[1]);
      }
      return new Layer(0, 0, TActivation.ReLU, 0);
    });
    this.outputValues = [];
    this.mse = [];
  }

  feedForward(inputValue: number[]) {
    this.layers[0].neurons.forEach((neuron, idx) => {
      neuron.value = inputValue[idx];
      neuron.activeValue = inputValue[idx];
    });
    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].fire(this.layers);
    }
    this.outputValues.push(
      this.layers[this.layers.length - 1].neurons.map((n) => n.activeValue)
    );
  }

  calcMse(testData: TTrainingData) {
    const mse = testData.reduce((prev, curr, idx) => {
      const sum = curr.label.reduce((prevv, currr, idxx) => {
        return prevv + Math.pow(currr - this.outputValues[idx][idxx], 2);
      }, 0);
      return prev + sum;
    }, 0);
    this.mse.push(mse);
  }

  train(trainingData: TTrainingData) {
    trainingData.forEach((sample) => {
      this.feedForward(sample.value);
      this.layers[1].backpropagate(sample, this.layers);
      this.layers[2].backpropagate(sample, this.layers);
      this.outputValues = [];
    });
    this.layers[1].update(trainingData.length);
    this.layers[2].update(trainingData.length);
  }
}
