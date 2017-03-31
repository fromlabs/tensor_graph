// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

import "package:tensor_graph/tensor_graph.dart";

var random = new Random();

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
      expecteds.add(toNum(xor(x1, x2)));
    }
  }
  return {"inputs": inputs, "expecteds": expecteds};
}

void main() {
  var watch = new Stopwatch();
  watch.start();

  // SOLVE XOR
  // y = x1 XOR x2

  var epochs = 100000;
  var learningRate = 0.01;

  new Session(new Model()).asDefault((session) {
    var x1 = new Reference(name: "x1");
    var x2 = new Reference(name: "x2");
    var expected = new Reference(name: "expected");

    var w11l1 = new Variable(nextDouble(-1, 1), name: "w11_l1");
    var w12l1 = new Variable(nextDouble(-1, 1), name: "w12_l1");
    var w21l1 = new Variable(nextDouble(-1, 1), name: "w21_l1");
    var w22l1 = new Variable(nextDouble(-1, 1), name: "w22_l1");
    var b1l1 = new Variable(0, name: "b1_l1");
    var b2l1 = new Variable(0, name: "b2_l1");

    var w1l2 = new Variable(nextDouble(-1, 1), name: "w1_l2");
    var w2l2 = new Variable(nextDouble(-1, 1), name: "w2_l2");
    var bl2 = new Variable(0, name: "b_l2");

    var logit1l1 = new Reference(
        defaultInput: w11l1 * x1 + w12l1 * x2 + b1l1, name: "logit1_l1");
    var logit2l1 = new Reference(
        defaultInput: w21l1 * x1 + w22l1 * x2 + b2l1, name: "logit2_l1");

    var output1l1 = new Sigmoid(logit1l1, name: "output1_l1");
    var output2l1 = new Sigmoid(logit2l1, name: "output2_l1");

    var logitl2 = new Reference(
        defaultInput: w1l2 * output1l1 + w2l2 * output2l1 + bl2,
        name: "logit_l2");

    var predicted = new Sigmoid(logitl2, name: "predicted");

    // var loss = new BinaryCrossEntropyLoss(expected, predicted);

    var loss = new BinaryCrossEntropyWithLogitLoss(expected, logitl2);

    var trainableVariables = [
      w11l1,
      w12l1,
      w21l1,
      w22l1,
      b1l1,
      b2l1,
      w1l2,
      w2l2,
      bl2
    ];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        checkingRate: 0.01,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var dataset = getDataset();

    for (var i in range(0, epochs)) {
      var lossAverage = 0.0;
      for (var entry in enumerate(dataset["inputs"])) {
        var l = entry.index;
        var inputValue = entry.element;
        var expectedValue = dataset["expecteds"][l];

        var values = session.runs([
          loss,
          optimizer
        ], feeds: {
          x1: inputValue[0],
          x2: inputValue[1],
          expected: expectedValue
        });

        lossAverage += values[loss];
      }
      lossAverage /= 4;

      if (i % 1000 == 0) {
        print("$i: $lossAverage");
      }

      if (lossAverage <= 0.05) {
        break;
      }

      if (i % 10000 == 0) {
        test(dataset, x1, x2, predicted, expected, loss, session);
      }
    }

    test(dataset, x1, x2, predicted, expected, loss, session);
  });
}

void test(Map<String, dynamic> dataset, Tensor x1, Tensor x2, Tensor predicted,
    Tensor expected, Tensor loss, Session session) {
  print("*** TEST ***");

  var lossAverage = 0.0;
  for (var entry in enumerate(dataset["inputs"])) {
    var i = entry.index;
    var inputValue = entry.element;
    var expectedValue = dataset["expecteds"][i];

    var values = session.runs([
      predicted,
      loss,
    ], feeds: {
      x1: inputValue[0],
      x2: inputValue[1],
      expected: expectedValue
    });

    lossAverage += values[loss];

    print("Input: $inputValue");
    print("Expected: $expectedValue");
    print("Predicted: ${values[predicted] >= 0.5} [${values[predicted]}]");
  }
  lossAverage /= 4;

  print("Loss: $lossAverage");
}
