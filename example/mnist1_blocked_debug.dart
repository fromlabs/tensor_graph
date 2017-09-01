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
  var dataset = await getDataset();

  var trainDataset = dataset["train"];

  var batchSize = 128;
  var learningRate = 0.00000005;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  var factor = 100;

  new tg.Session(new tg.Model()).asDefault((session) {
    var x = new tg.ModelInput(
        shapeDimensions: [null, 784],
        dataType: tm.NDDataType.float32Blocked,
        name: "x");
    var k = new tg.Constant(
        new tm.NDArray.generate(
            [784, 10], (index) => (random.nextDouble() - 0.5) / factor,
            dataType: tm.NDDataType.float32Blocked),
        dataType: tm.NDDataType.float32Blocked);
    var w = new tg.Variable(k, dataType: tm.NDDataType.float32Blocked);
    var k2 = new tg.Constant(
        new tm.NDArray.zeros([10], dataType: tm.NDDataType.float32Blocked),
        dataType: tm.NDDataType.float32Blocked);
    var b = new tg.Variable(k2, dataType: tm.NDDataType.float32Blocked);

    var y = new tg.MatMul(x, w) + b;

    var expected = new tg.ModelInput(
        shapeDimensions: [null, 10],
        dataType: tm.NDDataType.float32Blocked,
        name: "expected");

    var totalLoss = new tg.SoftmaxCrossEntropyWithLogits(expected, y);

    var loss = new tg.ReduceMean(totalLoss);

    var trainableVariables = [w, b];

    var optimizer = new tg.Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        name: "optimizer");
/*
    var sm = new tg.Softmax(y);

    var correctPrediction = new tg.IsEqual(
        new tg.ArgMax(sm, axis: 1), new tg.ArgMax(expected, axis: 1));

    var accuracy =
        new tg.ReduceMean(new tg.Select(correctPrediction, 1.0, 0.0));
*/
    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var indexes = generator.getBatchIndexes(batchSize);

    var imagesBatch = extractBatchFromIndexes(trainDataset["images"], indexes);
    var labelsBatch = extractBatchFromIndexes(trainDataset["labels"], indexes);

    var values = session.runs([loss, optimizer],
        feeds: {x: imagesBatch, expected: labelsBatch});

    print("loss = ${values[loss].toScalar()}");
  });
}
