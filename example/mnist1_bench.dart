// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart";

import "mnist_generator.dart" as mnist;
import "batch_generator.dart";

var random = new Random();

Future<Map<String, Map<String, List>>> getDataset() => mnist.createDataset();

Future main() async {
  var watch = new Stopwatch();
  watch.start();

  var dataset = await getDataset();

  var trainDataset = dataset["train"];

  var steps = 100;
  var batchSize = 128;
  var learningRate = 0.00000005;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  new Session(new Model()).asDefault((session) {
    var x = new ModelInput(shapeDimensions: [null, 784], name: "x");

    var w = new Variable(new NDArray.generate(
        [784, 10], (index) => (random.nextDouble() - 0.5) / 100));
    var b = new Variable(new NDArray.zeros([10], dataType: NDDataType.float32));
    var xw = new MatMul(x, w);

    var y = new Add(xw, b);

    var sm = new Softmax(y);

    var expected =
        new ModelInput(shapeDimensions: [null, 10], name: "expected");

    var loss = new ReduceMean(new SoftmaxCrossEntropyWithLogits(expected, y));

    var trainableVariables = [w, b];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    for (var i in range(1, steps)) {
      var indexes = generator.getBatchIndexes(batchSize);

      var imagesBatch =
          extractBatchFromIndexes(trainDataset["images"], indexes);
      var labelsBatch =
          extractBatchFromIndexes(trainDataset["labels"], indexes);

      var values = session.runs([loss, optimizer],
          feeds: {x: imagesBatch, expected: labelsBatch});

      print("Step $i: loss = ${values[loss].toScalar()}");
    }

    print("Finish in ${watch.elapsedMilliseconds} ms");
  });
}
