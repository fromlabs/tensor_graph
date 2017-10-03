// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart" as tg;
import "package:tensor_math/tensor_math.dart" as tm;

import "mnist_generator.dart" as mnist;
import "batch_generator.dart";

var random = new Random();

Future<Map<String, Map<String, List>>> getDataset() => mnist.createDataset();

Future main() async {
  var watch = new Stopwatch();
  watch.start();

  var dataset = await getDataset();

  var trainDataset = dataset["train"];
  var testDataset = dataset["test"];

  var steps = 10000;
  var batchSize = 128;
  //var learningRate = 0.00003;
  var learningRate = 0.0003;
  var checkStepInterval = 10;
  var testStepInterval = 1000;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  var factor = 10;

  new tg.Session(new tg.Model()).asDefault((session) {
    var x = new tg.ModelInput(shapeDimensions: [null, 784], name: "x");

    var x2 = new tg.Reshape(x, newDimensions: [-1, 28, 28, 1]);

    var x3 = new tg.Div(x2, 255);

    var w1 = new tg.Variable(new tm.NDArray.generate(
        [5, 5, 1, 32], (index) => (random.nextDouble() - 0.5) / factor));
    var b1 = new tg.Variable(
        new tm.NDArray.zeros([32], dataType: tm.NDDataType.float32));

    var conv =
        new tg.Relu(new tg.Convolution2d(x3, kernel: w1, name: "conv") + b1);

    var pool =
        new tg.MaxPool(conv, blockHeight: 2, blockWidth: 2, name: "pool");

    var flatten =
        new tg.Reshape(pool, newDimensions: [-1, 6272], name: "flatten");

    var w2 = new tg.Variable(new tm.NDArray.generate(
        [6272, 128], (index) => (random.nextDouble() - 0.5) / factor));
    var b2 = new tg.Variable(
        new tm.NDArray.zeros([128], dataType: tm.NDDataType.float32));

    var fc = new tg.Relu(new tg.MatMul(flatten, w2) + b2);

    var w = new tg.Variable(new tm.NDArray.generate(
        [128, 10], (index) => (random.nextDouble() - 0.5) / factor));
    var b = new tg.Variable(
        new tm.NDArray.zeros([10], dataType: tm.NDDataType.float32));

    var y = new tg.Relu(new tg.MatMul(fc, w) + b);

    var expected =
        new tg.ModelInput(shapeDimensions: [null, 10], name: "expected");

    var loss =
        new tg.ReduceMean(new tg.SoftmaxCrossEntropyWithLogits(expected, y));

    var trainableVariables = [
      w1,
      b1,
      w2,
      b2,
      w,
      b
    ]; // TODO variabili della convoluzione

    var optimizer = new tg.Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        name: "optimizer");

    var sm = new tg.Softmax(y);

    var correctPrediction = new tg.IsEqual(
        new tg.ArgMax(sm, axis: 1), new tg.ArgMax(expected, axis: 1));

    var accuracy =
        new tg.ReduceMean(new tg.Select(correctPrediction, 1.0, 0.0));

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var previousCheck = watch.elapsedMilliseconds;
    for (var i in tg.range(0, steps)) {
      var indexes = generator.getBatchIndexes(batchSize);

      var imagesBatch =
          extractBatchFromIndexes(trainDataset["images"], indexes);
      var labelsBatch =
          extractBatchFromIndexes(trainDataset["labels"], indexes);
/*
      print(session.run(y0,
          feeds: {x: imagesBatch, expected: labelsBatch}));
*/

      var values = session.runs([loss, optimizer, accuracy],
          feeds: {x: imagesBatch, expected: labelsBatch});

      if (i > 0 && i % checkStepInterval == 0) {
        var throughput = 1000 *
            checkStepInterval /
            (watch.elapsedMilliseconds - previousCheck);
        previousCheck = watch.elapsedMilliseconds;

        print(
            "Step $i: loss = ${values[loss].toScalar()}, accuracy = ${values[accuracy].toScalar()} [$throughput step/sec]");
      }

      if (i > 0 && i % testStepInterval == 0) {
        test(testDataset, x, y, expected, loss, correctPrediction, accuracy,
            session);
      }
    }

    test(testDataset, x, y, expected, loss, correctPrediction, accuracy,
        session);

    print("Finish in ${watch.elapsedMilliseconds} ms");
  });
}

void test(
    Map<String, dynamic> dataset,
    tg.Tensor x,
    tg.Tensor y,
    tg.Tensor expected,
    tg.Tensor loss,
    tg.Tensor correctPrediction,
    tg.Tensor accuracy,
    tg.Session session) {
  print("*** TEST ***");

  var values = session.runs([loss, correctPrediction, accuracy],
      feeds: {x: dataset["images"], expected: dataset["labels"]});

  print("Loss: ${values[loss].toScalar()}");
  print("Accuracy: ${values[accuracy].toScalar() * 100} %");
}
