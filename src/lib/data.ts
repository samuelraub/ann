export const compareFunction = ([a, b]: number[]) =>
  Math.max(Math.pow(a, 2), -Math.pow(2 * a, 2) + 0.5) > b;
//  Math.sin(10 * a) / 4 + 0.5 > b;
// Math.pow(2 * a - 1, 2) > b;
// a < 0.5 && b < 0.5;
// a + 0.2 < b;

export const testData = new Array(500)
  .fill(new Array(2).fill(0))
  .map((arr) => arr.map(() => Math.random()));

export const testDataWithLabels = testData.map((vals) => ({
  value: vals,
  label: compareFunction(vals) ? [1, 0] : [0, 1]
}));

export const trainingData = new Array(100000)
  .fill(new Array(2).fill(0))
  .map((arr) => arr.map(() => Math.random()))
  .map((vals) => ({
    value: vals,
    label: compareFunction(vals) ? [1, 0] : [0, 1]
  }));
