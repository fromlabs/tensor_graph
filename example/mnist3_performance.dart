// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart";

import "mnist_generator.dart" as mnist;
import "batch_generator.dart";

/*

*** TEST ***
Loss: 0.09449993460713639
Accuracy: 97.28 %
Finish in 5578144 ms

 */

var random = new Random();

Future<Map<String, Map<String, List>>> getDataset() {
  return mnist.createDataset();
}

Future main() async {
  var dataset = await getDataset();

  var trainDataset = dataset["train"];
  var testDataset = dataset["test"];

  var steps = 1000;
  var batchSize = 128;
  var learningRate = 0.00003;
  var checkStepInterval = 10;
  var testStepInterval = 1000;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  var l0 = 200;
  var l1 = 100;
  var l2 = 60;
  var l3 = 30;
  var factor = 10;

  new Session(new Model()).asDefault((session) {
    var x = new ModelInput(shapeDimensions: [null, 784], name: "x");
    var w0 = new Variable(new NDArray.generate(
        [784, l0], (index) => (random.nextDouble() - 0.5) / factor));
    var b0 = new Variable(new NDArray.zeros([l0]));
    var w1 = new Variable(new NDArray.generate(
        [l0, l1], (index) => (random.nextDouble() - 0.5) / factor));
    var b1 = new Variable(new NDArray.zeros([l1]));
    var w2 = new Variable(new NDArray.generate(
        [l1, l2], (index) => (random.nextDouble() - 0.5) / factor));
    var b2 = new Variable(new NDArray.zeros([l2]));
    var w3 = new Variable(new NDArray.generate(
        [l2, l3], (index) => (random.nextDouble() - 0.5) / factor));
    var b3 = new Variable(new NDArray.zeros([l3]));
    var w = new Variable(new NDArray.generate(
        [l3, 10], (index) => (random.nextDouble() - 0.5) / factor));
    var b = new Variable(new NDArray.zeros([10]));

    var y0 = new Relu(new MatMul(x, w0) + b0);
    var y1 = new Relu(new MatMul(y0, w1) + b1);
    var y2 = new Relu(new MatMul(y1, w2) + b2);
    var y3 = new Relu(new MatMul(y2, w3) + b3);

    var y = new MatMul(y3, w) + b;

    var expected =
        new ModelInput(shapeDimensions: [null, 10], name: "expected");

    var loss = new ReduceMean(new SoftmaxCrossEntropyWithLogits(expected, y));

    var trainableVariables = [w0, b0, w1, b1, w2, b2, w3, b3, w, b];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        name: "optimizer");

    var sm = new Softmax(y);

    var correctPrediction =
        new IsEqual(new ArgMax(sm, axis: 1), new ArgMax(expected, axis: 1));

    var accuracy = new ReduceMean(new Select(correctPrediction, 1, 0));

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var watch = new Stopwatch();
    watch.start();

    print("Start...");

    for (var i in range(0, steps)) {
      var indexes = generator.getBatchIndexes(batchSize);

      var imagesBatch =
          extractBatchFromIndexes(trainDataset["images"], indexes);
      var labelsBatch =
          extractBatchFromIndexes(trainDataset["labels"], indexes);

      session.runs([loss, optimizer],
          feeds: {x: imagesBatch, expected: labelsBatch});
    }

    print("Finish in ${watch.elapsedMilliseconds} ms");
  });
}

void test(Map<String, dynamic> dataset, Tensor x, Tensor y, Tensor expected,
    Tensor loss, Tensor correctPrediction, Tensor accuracy, Session session) {
  print("*** TEST ***");

  var values = session.runs([loss, correctPrediction, accuracy],
      feeds: {x: dataset["images"], expected: dataset["labels"]});

  print("Loss: ${values[loss].toScalar()}");
  print("Accuracy: ${values[accuracy].toScalar() * 100} %");
}
