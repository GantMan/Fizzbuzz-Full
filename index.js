import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";

import "./styles.css";

const inputShape = 10;

// cleanup if running too many at once
//const preExistingBackend = tf.getBackend();
//if (preExistingBackend) tf.removeBackend(preExistingBackend);

const numToBinTensor = num =>
  tf.tensor(
    num
      .toString(2)
      .padStart(inputShape, "0")
      .split("")
      .map(Number) // DO NOT USE parseInt here
  );

const fizzbuzzEncoder = num => {
  if (num % 15 === 0) {
    return tf.oneHot(3, 4);
  } else if (num % 5 === 0) {
    return tf.oneHot(2, 4);
  } else if (num % 3 === 0) {
    return tf.oneHot(1, 4);
  } else {
    return tf.oneHot(0, 4);
  }
};

// Wrap in a tidy for memory
const [stackedX, stackedY] = tf.tidy(() => {
  let xs = [];
  let ys = [];
  for (let i = 1; i <= 1000; i++) {
    xs.push(numToBinTensor(i));
    ys.push(fizzbuzzEncoder(i));
  }

  return [tf.stack(xs), tf.stack(ys)];
});

const doLinearPrediction = async () => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 64,
      inputShape: inputShape,
      activation: "relu"
    })
  );

  model.add(
    tf.layers.dense({
      units: 8,
      activation: "relu"
    })
  );

  model.add(
    tf.layers.dense({
      units: 4,
      kernelInitializer: "varianceScaling",
      activation: "softmax"
    })
  );

  const learningRate = 0.005;
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  // Make loss callback
  const printCallback = {
    onEpochEnd: (epoch, log) => {
      console.log(log);
    }
  };

  console.log("starting fit");
  await model.fit(stackedX, stackedY, {
    epochs: 100,
    shuffle: true,
    batchSize: 32,
    callbacks: printCallback
  });

  console.log("done");

  // ******************************
  // Debug check
  // *****************************
  // const next = tf.stack([
  //   numToBinTensor(1),
  //   numToBinTensor(3),
  //   numToBinTensor(5),
  //   numToBinTensor(15)
  // ]);
  // const answer = model.predict(next);
  // answer.print();

  const fizzBuzzResult = [];
  for (let x = 1; x < 100; x++) {
    const singlePredictInput = numToBinTensor(x).reshape([1, 10]);
    const resultData = await model.predict(singlePredictInput).data();
    // grab Max index
    const winner = resultData.indexOf(Math.max(...resultData));
    const result = [x, "fizz", "buzz", "fizzbuzz"][winner];
    // console.log(result);
    fizzBuzzResult.push(result);
    singlePredictInput.dispose(); // manual dispose
  }

  // for save button
  window.model = model;
  return fizzBuzzResult.join(", ");
};

class App extends React.Component {
  state = {
    simplePredict: "training model..."
  };

  componentDidMount() {
    doLinearPrediction().then(result =>
      this.setState({ simplePredict: result })
    );
  }

  render() {
    return (
      <div className="App">
        <h1>FizzBuzz ML</h1>
        <img src="https://i.imgur.com/YrUqtM0.png" width="150" />
        <h3>
          <a href="http://gantlaborde.com/">By Gant Laborde</a>
        </h3>
        <h2>{this.state.simplePredict}</h2>
        <button
          onClick={async () => {
            if (!window.model) return;
            await window.model.save("downloads://fizzbuzz-model");
          }}
        >
          Download Resulting Model
        </button>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
