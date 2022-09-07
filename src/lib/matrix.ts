export default class Matrix {
  numberArray: number[][];

  constructor(numberArray: number[][]) {
    this.numberArray = numberArray;
  }

  private checkDimensions(m1: Matrix, m2: Matrix, func: string) {
    if (func === "add") {
      return (
        m1.numberArray.length === m2.numberArray.length &&
        m1.numberArray[0].length === m2.numberArray[0].length
      );
    }
    if (func === "multiply") {
      return (
        m1.numberArray[0].length === m2.numberArray.length &&
        m1.numberArray.length === m2.numberArray[0].length
      );
    }
  }

  add(otherMatrix: Matrix): Matrix {
    if (!this.checkDimensions(this, otherMatrix, "add")) {
      throw new Error("dimension mismatch");
    }
    const numberArray = this.numberArray.map((row, rowIdx) =>
      row.map((col, colIdx) => {
        return col + otherMatrix.numberArray[rowIdx][colIdx];
      })
    );
    return new Matrix(numberArray);
  }

  subtract(otherMatrix: Matrix): Matrix {
    if (!this.checkDimensions(this, otherMatrix, "add")) {
      throw new Error("dimension mismatch");
    }
    const numberArray = this.numberArray.map((row, rowIdx) =>
      row.map((col, colIdx) => {
        return col - otherMatrix.numberArray[rowIdx][colIdx];
      })
    );
    return new Matrix(numberArray);
  }

  multiply(otherMatrix: Matrix): Matrix {
    if (!this.checkDimensions(this, otherMatrix, "multiply")) {
      throw new Error("dimension mismatch");
    }
    let result = [];
    for (let row = 0; row < this.numberArray.length; row++) {
      let newRow = [];
      for (let col = 0; col < this.numberArray.length; col++) {
        let sum = 0;
        for (let idx = 0; idx < this.numberArray[0].length; idx++) {
          sum += this.numberArray[row][idx] * otherMatrix.numberArray[idx][col];
        }
        newRow.push(sum);
      }
      result.push(newRow);
    }
    return new Matrix(result);
  }

  print() {
    console.table(this.numberArray);
  }
}
