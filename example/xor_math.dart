// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart" as math;

var random = new Random(0);

num nextDouble(num from, num to) => random.nextDouble() * (to - from) + from;

bool xor(bool x1, bool x2) => (x1 && !x2) || (!x1 && x2);

num toNum(bool x, {num trueValue = 1, num falseValue = 0}) =>
    (x ? 1 : 0) * (trueValue - falseValue) + falseValue;

Map<String, dynamic> getDataset() {
  var inputs = [];
  var expecteds = [];
  for (var x1 in [false, true]) {
    for (var x2 in [false, true]) {
      inputs.add([toNum(x1), toNum(x2)]);
      expecteds.add([toNum(xor(x1, x2))]);
    }
  }
  return {"inputs": inputs, "expecteds": expecteds};
}

void main() {
  var watch = new Stopwatch();
  watch.start();

  var learningRate = 0.01;

  new Session(new Model()).asDefault((session) {
    var x = new Reference(shape: [null, 2], name: "x");
    var expected = new Reference(shape: [null, 1], name: "expected");

    var wl1 = new Variable([
      [nextDouble(-1, 1), nextDouble(-1, 1)],
      [nextDouble(-1, 1), nextDouble(-1, 1)]
    ], name: "w_l1");
    var bl1 = new Variable([0, 0], name: "b_l1");

    var wl2 = new Variable([
      [nextDouble(-1, 1)],
      [nextDouble(-1, 1)]
    ], name: "w_l2");
    var bl2 = new Variable(0, name: "b_l2");

    var trainableVariables = [wl1, bl1, wl2, bl2];

    var logitl1 =
        new Reference(target: new MatMul(x, wl1) + bl1, name: "logit_l1");

    var outputl1 = new Sigmoid(logitl1, name: "output_l1");

    var logitl2 = new Reference(
        target: new MatMul(outputl1, wl2) + bl2, name: "logit_l2");

    var predicted = new Sigmoid(logitl2, name: "predicted");

    var loss = new Mean(new SigmoidCrossEntropyLoss(expected, predicted),
        name: "loss");

    // var loss = new Mean(new SigmoidCrossEntropyWithLogitLoss(expected, logitl2), name: "loss");

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        checkingRate: 1,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var dataset = getDataset();

    var value = session.run(optimizer,
        feeds: {x: dataset["inputs"], expected: dataset["expecteds"]});

    print("value: $value [shape: ${math.shape(value)}]");
  });
}
