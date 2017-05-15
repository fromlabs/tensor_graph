// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart";

import "mnist_generator.dart" as mnist;
import "batch_generator.dart";

var random = new Random();

Future<Map<String, Map<String, List>>> getDataset() {
  return mnist.createDataset();
}

Future main() async {
  var watch = new Stopwatch();
  watch.start();

  var dataset = await getDataset();

  var trainDataset = dataset["train"];
  var testDataset = dataset["test"];

  var steps = 10000;
  var batchSize = 128;
  var learningRate = 0.0000005;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  var l0 = 10;

  new Session(new Model()).asDefault((session) {
    var x = new Placeholder(shapeDimensions: [null, 784], name: "x");
    // var w0 = new Variable(new NDArray.zeros([784, l0]));
    var w0 = new Variable(new NDArray.generate(
        [784, l0], (index) => (random.nextDouble() - 0.5) / 100));
    var b0 = new Variable(new NDArray.zeros([l0]));
    // var w = new Variable(new NDArray.zeros([l0, 10]));
    var w = new Variable(new NDArray.generate(
        [l0, 10], (index) => (random.nextDouble() - 0.5) / 100));
    var b = new Variable(new NDArray.zeros([10]));

    var y0 = new Relu(new MatMul(x, w0) + b0);

    var y = new MatMul(y0, w) + b;

    var expected = new Placeholder(shapeDimensions: [null, 10], name: "expected");

    var loss = new ReduceMean(new SoftmaxCrossEntropyWithLogits(expected, y));

    var trainableVariables = [w0, b0, w, b];

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

    for (var i in range(0, steps)) {
      var indexes = generator.getBatchIndexes(batchSize);

      var imagesBatch =
          extractBatchFromIndexes(trainDataset["images"], indexes);
      var labelsBatch =
          extractBatchFromIndexes(trainDataset["labels"], indexes);
/*
      print(session.run(y0,
          feeds: {x: imagesBatch, expected: labelsBatch}));
*/
      var values = session.runs([loss, optimizer],
          feeds: {x: imagesBatch, expected: labelsBatch});

      if (i % 10 == 0) {
        print("Step $i: loss = ${values[loss].toScalar()}");
      }
    }

    test(testDataset, x, y, expected, loss, correctPrediction, accuracy,
        session);

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
