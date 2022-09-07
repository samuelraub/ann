import { useState } from "react";
import _ from "lodash";
import Network from "./lib/network";
import { trainingData, testDataWithLabels } from "./lib/data";
import { LineChart, Line } from "recharts";
import Matrix from "./lib/matrix";

const EPOCHS = 100;
const BATCH_SIZE = 100;
const STEPS = 100;

const net = new Network([2, 4, 2]);

const App = () => {
  const [outputValues, setOutputValues] = useState<number[][]>([]);
  const [mse, setMse] = useState<number[]>([]);

  return (
    <>
      <button
        onClick={() => {
          for (let i = 0; i < STEPS; i++) {
            for (let i = 0; i < EPOCHS; i++) {
              net.train(_.sampleSize(trainingData, BATCH_SIZE));
            }
            testDataWithLabels.forEach((sample) =>
              net.feedForward(sample.value)
            );
            net.calcMse(testDataWithLabels);
            setMse(net.mse);
            setOutputValues(net.outputValues);
            net.outputValues = [];
          }
        }}
      >
        Train
      </button>
      <div style={{ position: "relative", height: "600px", width: "600px" }}>
        {testDataWithLabels.map(({ value, label }, idx) => (
          <div
            key={idx}
            style={{
              borderRadius: "999px",
              background: label[0] > label[1] ? "slateblue" : "darkseagreen",
              position: "absolute",
              left: value[0] * 500 + "px",
              top: value[1] * 500 + "px",
              width: "10px",
              height: "10px"
            }}
          />
        ))}
        {outputValues.map((data, idx) => (
          <div
            key={idx}
            style={{
              borderRadius: "999px",
              border:
                "5px solid " +
                (data[0] > data[1] ? "slateblue" : "darkseagreen"),
              position: "absolute",
              left: testDataWithLabels[idx].value[0] * 500 - 5 + "px",
              top: testDataWithLabels[idx].value[1] * 500 - 5 + "px",
              width: "10px",
              height: "10px"
            }}
            onClick={() => console.log(data)}
          />
        ))}
      </div>
      <div style={{ border: "1px solid black" }}>
        <LineChart
          data={mse.map((el) => ({ mse: el }))}
          width={400}
          height={400}
        >
          <Line type="monotone" dataKey={"mse"} stroke="#8884d8" />
        </LineChart>
      </div>
    </>
  );
};

export default App;
